"""
Authentication Router - Login, logout, session management.

Provides:
- Email/password login
- Session-based authentication
- Optional registration
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, EmailStr

from gateway.logging_setup import log
from gateway import config


router = APIRouter(prefix="/auth", tags=["auth"])


# =============================================================================
# Pydantic Models
# =============================================================================

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    email: str
    display_name: Optional[str]
    role: str


# =============================================================================
# Password Hashing
# =============================================================================

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    try:
        import bcrypt
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    except ImportError:
        # Fallback to hashlib if bcrypt not available
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    try:
        import bcrypt
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ImportError:
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest() == password_hash


# =============================================================================
# Session Management
# =============================================================================

def generate_session_token() -> str:
    """Generate a secure session token."""
    return secrets.token_urlsafe(32)


async def create_session(
    user_id: int,
    request: Request,
) -> str:
    """Create a new session for a user."""
    from gateway.db import get_session
    from sqlalchemy import text
    
    token = generate_session_token()
    expires_at = datetime.utcnow() + timedelta(hours=config.SESSION_EXPIRY_HOURS)
    
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent", "")[:500]
    
    async with get_session() as session:
        await session.execute(
            text("""
                INSERT INTO sessions (user_id, token, expires_at, ip_address, user_agent)
                VALUES (:user_id, :token, :expires_at, :ip_address, :user_agent)
            """),
            {
                "user_id": user_id,
                "token": token,
                "expires_at": expires_at,
                "ip_address": ip_address,
                "user_agent": user_agent,
            },
        )
    
    return token


async def get_session_user(token: str) -> Optional[dict]:
    """Get user from session token."""
    from gateway.db import get_session
    from sqlalchemy import text
    
    async with get_session() as session:
        result = await session.execute(
            text("""
                SELECT u.id, u.email, u.display_name, u.role
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.token = :token
                AND s.expires_at > NOW()
                AND u.is_active = true
            """),
            {"token": token},
        )
        row = result.fetchone()
        
        if row:
            return {
                "id": row[0],
                "email": row[1],
                "display_name": row[2],
                "role": row[3],
            }
    
    return None


async def delete_session(token: str):
    """Delete a session."""
    from gateway.db import get_session
    from sqlalchemy import text
    
    async with get_session() as session:
        await session.execute(
            text("DELETE FROM sessions WHERE token = :token"),
            {"token": token},
        )


# =============================================================================
# Auth Dependencies
# =============================================================================

async def get_current_user(request: Request) -> Optional[dict]:
    """Get current user from session cookie."""
    token = request.cookies.get("session")
    if not token:
        return None
    
    return await get_session_user(token)


async def require_auth(request: Request) -> dict:
    """Require authentication - raises 401 if not authenticated."""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


async def require_admin(request: Request) -> dict:
    """Require admin role."""
    user = await require_auth(request)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# =============================================================================
# Routes
# =============================================================================

@router.post("/login")
async def login(request: Request, body: LoginRequest, response: Response):
    """
    Login with email and password.
    Returns session cookie on success.
    """
    from gateway.db import get_session
    from sqlalchemy import text
    
    async with get_session() as session:
        result = await session.execute(
            text("""
                SELECT id, password_hash, is_active
                FROM users
                WHERE email = :email
            """),
            {"email": body.email},
        )
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        user_id, password_hash, is_active = row
        
        if not is_active:
            raise HTTPException(status_code=401, detail="Account disabled")
        
        if not verify_password(body.password, password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Update last login
        await session.execute(
            text("UPDATE users SET last_login = NOW() WHERE id = :id"),
            {"id": user_id},
        )
    
    # Create session
    token = await create_session(user_id, request)
    
    # Set cookie
    response.set_cookie(
        key="session",
        value=token,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        max_age=config.SESSION_EXPIRY_HOURS * 3600,
    )
    
    log.info("User logged in: %s", body.email)
    
    return {"success": True, "message": "Logged in"}


@router.post("/logout")
async def logout(request: Request, response: Response):
    """Logout and invalidate session."""
    token = request.cookies.get("session")
    
    if token:
        await delete_session(token)
    
    response.delete_cookie("session")
    
    return {"success": True, "message": "Logged out"}


@router.get("/me")
async def get_me(user: dict = Depends(require_auth)):
    """Get current authenticated user."""
    return UserResponse(**user)


@router.post("/register")
async def register(body: RegisterRequest):
    """
    Register a new user.
    Only available if ALLOW_REGISTRATION is enabled.
    """
    if not config.ALLOW_REGISTRATION:
        raise HTTPException(status_code=403, detail="Registration disabled")
    
    from gateway.db import get_session
    from sqlalchemy import text
    
    password_hash = hash_password(body.password)
    
    try:
        async with get_session() as session:
            result = await session.execute(
                text("""
                    INSERT INTO users (email, password_hash, display_name, role)
                    VALUES (:email, :password_hash, :display_name, 'user')
                    RETURNING id
                """),
                {
                    "email": body.email,
                    "password_hash": password_hash,
                    "display_name": body.display_name or body.email.split("@")[0],
                },
            )
            user_id = result.scalar()
        
        log.info("New user registered: %s", body.email)
        
        return {"success": True, "user_id": user_id}
        
    except Exception as e:
        if "unique" in str(e).lower():
            raise HTTPException(status_code=400, detail="Email already registered")
        raise


# =============================================================================
# Login Page HTML
# =============================================================================

LOGIN_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login | AI Gateway</title>
    <style>
        :root {
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --primary: #38bdf8;
            --error: #ef4444;
            --border: #334155;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 2rem;
            width: 100%;
            max-width: 400px;
        }
        h1 {
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            text-align: center;
            background: linear-gradient(to right, var(--primary), #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .form-group {
            margin-bottom: 1rem;
        }
        label {
            display: block;
            font-size: 0.875rem;
            color: var(--text-dim);
            margin-bottom: 0.5rem;
        }
        input {
            width: 100%;
            padding: 0.75rem;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            color: var(--text);
            font-size: 1rem;
        }
        input:focus {
            outline: none;
            border-color: var(--primary);
        }
        button {
            width: 100%;
            padding: 0.75rem;
            background: var(--primary);
            color: #000;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            margin-top: 1rem;
        }
        button:hover { opacity: 0.9; }
        .error {
            color: var(--error);
            font-size: 0.875rem;
            margin-top: 1rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="login-card">
        <h1>AI Gateway</h1>
        <form id="loginForm">
            <div class="form-group">
                <label for="email">Email</label>
                <input type="email" id="email" name="email" required>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit">Login</button>
            <div id="error" class="error"></div>
        </form>
    </div>
    <script>
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const errorEl = document.getElementById('error');
            
            try {
                const resp = await fetch('/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password }),
                });
                
                if (resp.ok) {
                    window.location.href = '/dashboard';
                } else {
                    const data = await resp.json();
                    errorEl.textContent = data.detail || 'Login failed';
                }
            } catch (err) {
                errorEl.textContent = 'Connection error';
            }
        });
    </script>
</body>
</html>
"""


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve login page."""
    # If already logged in, redirect to dashboard
    user = await get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=302)
    
    return HTMLResponse(content=LOGIN_PAGE_HTML)
