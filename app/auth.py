"""Login sessions (HttpOnly cookie) and role checks for the API and the pages."""
from fastapi import Cookie, HTTPException

from . import db

COOKIE = "rg_session"


def current_user(rg_session: str | None = Cookie(default=None)):
    return db.session_user(rg_session)


def require_user(rg_session: str | None = Cookie(default=None)) -> dict:
    user = db.session_user(rg_session)
    if not user:
        raise HTTPException(401, "Sign in first.")
    return user


def require_customer(rg_session: str | None = Cookie(default=None)) -> dict:
    user = require_user(rg_session)
    if user["role"] != "customer":
        raise HTTPException(403, "This page is for customers.")
    return user


def require_reviewer(rg_session: str | None = Cookie(default=None)) -> dict:
    user = require_user(rg_session)
    if user["role"] != "reviewer":
        raise HTTPException(403, "This page is for reviewers.")
    return user


def require_courier(rg_session: str | None = Cookie(default=None)) -> dict:
    user = require_user(rg_session)
    if user["role"] != "courier":
        raise HTTPException(403, "This page is for pickup partners.")
    return user
