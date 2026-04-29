"""Pydantic models for telegram-fetcher.

Field list follows architecture spec section 2. Kept permissive: all datetimes
are timezone-aware (UTC) and serialize as ISO 8601.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


AuthStateLiteral = Literal["disconnected", "awaiting_code", "awaiting_2fa", "authenticated"]
JobStatusLiteral = Literal["queued", "in_progress", "done", "failed", "cancelled"]


class MediaItem(BaseModel):
    channel_id: str
    message_id: int
    caption: Optional[str] = None
    filename: Optional[str] = None
    size_bytes: int = 0
    mime_type: Optional[str] = None
    date: datetime
    author: Optional[str] = None
    title: Optional[str] = None
    chapter_num: Optional[int] = None
    manual_metadata: bool = False
    available: bool = True
    downloaded_path: Optional[str] = None
    thumbnail_path: Optional[str] = None


class InstructionalGroup(BaseModel):
    author: str
    title: str
    chapter_count: int
    total_size_bytes: int
    available: bool
    message_ids: List[int] = Field(default_factory=list)
    downloaded_chapters: int = 0
    # First-chapter preview (chapter_num=1 preferred, lowest message_id fallback).
    first_channel_id: Optional[str] = None
    first_message_id: Optional[int] = None
    first_thumbnail_path: Optional[str] = None


class AuthorView(BaseModel):
    name: str
    instructionals: List[InstructionalGroup] = Field(default_factory=list)


class DownloadJob(BaseModel):
    id: str
    channel_id: str
    author: str
    title: str
    message_ids: List[int]
    status: JobStatusLiteral = "queued"
    current_index: int = 0
    total: int
    current_pct: float = 0.0
    overall_pct: float = 0.0
    destination_dir: str
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class AuthState(BaseModel):
    state: AuthStateLiteral = "disconnected"
    phone: Optional[str] = None
    phone_code_hash: Optional[str] = None
    session_age_s: Optional[int] = None
    me_username: Optional[str] = None


class SyncReport(BaseModel):
    channel_id: str
    scanned: int = 0
    new: int = 0
    updated: int = 0
    parser_hits: int = 0
    parser_misses: int = 0
    elapsed_s: float = 0.0
