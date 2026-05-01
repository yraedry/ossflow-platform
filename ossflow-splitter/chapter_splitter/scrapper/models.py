from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class OracleChapter(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)

    @model_validator(mode="after")
    def _end_after_start(self) -> "OracleChapter":
        if self.end_s <= self.start_s:
            raise ValueError(
                f"end_s ({self.end_s}) must be > start_s ({self.start_s})"
            )
        return self

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class OracleVolume(BaseModel):
    number: int = Field(ge=1, le=50)
    chapters: list[OracleChapter] = Field(min_length=1)
    total_duration_s: float = Field(gt=0)

    @field_validator("chapters")
    @classmethod
    def _monotonic_starts(
        cls, v: list[OracleChapter]
    ) -> list[OracleChapter]:
        for i in range(1, len(v)):
            if v[i].start_s < v[i - 1].start_s:
                raise ValueError(
                    f"chapter {i} start_s={v[i].start_s} < "
                    f"previous start_s={v[i - 1].start_s}"
                )
        return v


class OracleResult(BaseModel):
    product_url: str
    provider_id: str
    scraped_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    poster_url: Optional[str] = None
    volumes: list[OracleVolume] = Field(min_length=1)

    @field_validator("volumes")
    @classmethod
    def _unique_volume_numbers(
        cls, v: list[OracleVolume]
    ) -> list[OracleVolume]:
        nums = [vol.number for vol in v]
        if len(nums) != len(set(nums)):
            raise ValueError(f"duplicate volume numbers: {nums}")
        return v

    def volume(self, number: int) -> Optional[OracleVolume]:
        for vol in self.volumes:
            if vol.number == number:
                return vol
        return None


class Candidate(BaseModel):
    url: str
    title: str
    author: Optional[str] = None
    score: float = Field(ge=0.0, le=1.0)
    provider_id: str
