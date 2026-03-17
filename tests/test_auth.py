"""Tests for authentication endpoints: /api/auth."""

from unittest.mock import patch

import pytest


class TestRegister:
    def test_register_user(self, client):
        payload = {"email": "newuser@example.com", "password": "securepass123"}
        resp = client.post("/api/auth/register", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "newuser@example.com"
        assert "id" in data
        assert data["is_active"] is True

    def test_register_duplicate_email(self, client):
        payload = {"email": "dup@example.com", "password": "securepass123"}
        resp1 = client.post("/api/auth/register", json=payload)
        assert resp1.status_code == 200

        resp2 = client.post("/api/auth/register", json=payload)
        assert resp2.status_code == 400
        assert "already registered" in resp2.json()["detail"]

    def test_register_short_password(self, client):
        payload = {"email": "short@example.com", "password": "short"}
        resp = client.post("/api/auth/register", json=payload)
        assert resp.status_code == 400
        assert "8 characters" in resp.json()["detail"]

    def test_first_user_becomes_admin(self, client):
        payload = {"email": "admin@example.com", "password": "adminpass123"}
        resp = client.post("/api/auth/register", json=payload)
        assert resp.status_code == 200
        assert resp.json()["role"] == "admin"

    def test_second_user_is_regular(self, client):
        client.post("/api/auth/register", json={"email": "first@x.com", "password": "password123"})
        resp = client.post("/api/auth/register", json={"email": "second@x.com", "password": "password123"})
        assert resp.status_code == 200
        assert resp.json()["role"] == "user"


class TestLogin:
    def test_login_success(self, client, test_user):
        resp = client.post(
            "/api/auth/login",
            data={"username": "testuser@example.com", "password": "testpassword123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client, test_user):
        resp = client.post(
            "/api/auth/login",
            data={"username": "testuser@example.com", "password": "wrongpassword"},
        )
        assert resp.status_code == 401
        assert "Incorrect" in resp.json()["detail"]

    def test_login_nonexistent_user(self, client):
        resp = client.post(
            "/api/auth/login",
            data={"username": "nobody@example.com", "password": "whatever123"},
        )
        assert resp.status_code == 401

    def test_login_email_case_insensitive(self, client, test_user):
        resp = client.post(
            "/api/auth/login",
            data={"username": "TestUser@Example.com", "password": "testpassword123"},
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()


class TestRefreshToken:
    def test_refresh_token(self, client, test_user):
        # First login to get a refresh token
        login_resp = client.post(
            "/api/auth/login",
            data={"username": "testuser@example.com", "password": "testpassword123"},
        )
        refresh_tok = login_resp.json()["refresh_token"]

        resp = client.post("/api/auth/refresh", json={"refresh_token": refresh_tok})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_refresh_with_invalid_token(self, client):
        resp = client.post("/api/auth/refresh", json={"refresh_token": "invalid.token.here"})
        assert resp.status_code == 401

    def test_refresh_with_access_token_rejected(self, client, test_user):
        """Using an access token as a refresh token should fail."""
        resp = client.post("/api/auth/refresh", json={"refresh_token": test_user["token"]})
        assert resp.status_code == 401


class TestProtectedEndpoint:
    def test_me_without_token(self, client):
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401

    def test_me_with_valid_token(self, client, test_user):
        resp = client.get("/api/auth/me", headers=test_user["headers"])
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "testuser@example.com"

    def test_me_with_invalid_token(self, client):
        resp = client.get("/api/auth/me", headers={"Authorization": "Bearer invalid.jwt.token"})
        assert resp.status_code == 401
