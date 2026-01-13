# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for TTL functionality in DatabaseSessionService."""

from __future__ import annotations

import asyncio
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from unittest import mock

from google.adk.sessions.database_session_service import DatabaseSessionService
import pytest


@pytest.fixture
async def ttl_session_service():
  """Creates a DatabaseSessionService with TTL enabled."""
  service = DatabaseSessionService(
      'sqlite+aiosqlite:///:memory:',
      ttl_seconds=60  # 60 seconds TTL
  )
  yield service
  await service.close()


@pytest.fixture
async def no_ttl_session_service():
  """Creates a DatabaseSessionService without TTL."""
  service = DatabaseSessionService('sqlite+aiosqlite:///:memory:')
  yield service
  await service.close()


@pytest.mark.asyncio
async def test_ttl_initialization():
  """Test that TTL can be set during initialization."""
  service = DatabaseSessionService(
      'sqlite+aiosqlite:///:memory:',
      ttl_seconds=120
  )
  assert service.ttl_seconds == 120
  await service.close()


@pytest.mark.asyncio
async def test_no_ttl_initialization():
  """Test that TTL defaults to None."""
  service = DatabaseSessionService('sqlite+aiosqlite:///:memory:')
  assert service.ttl_seconds is None
  await service.close()


@pytest.mark.asyncio
async def test_get_session_excludes_expired(ttl_session_service):
  """Test that get_session returns None for expired sessions."""
  app_name = 'test_app'
  user_id = 'test_user'
  
  # Create a session
  session = await ttl_session_service.create_session(
      app_name=app_name,
      user_id=user_id
  )
  session_id = session.id
  
  # Verify session exists
  retrieved = await ttl_session_service.get_session(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id
  )
  assert retrieved is not None
  assert retrieved.id == session_id
  
  # Mock the update_time to be older than TTL
  with mock.patch.object(
      ttl_session_service,
      '_is_session_expired',
      return_value=True
  ):
    # Session should now appear expired
    retrieved = await ttl_session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    assert retrieved is None


@pytest.mark.asyncio
async def test_list_sessions_excludes_expired(ttl_session_service):
  """Test that list_sessions excludes expired sessions."""
  app_name = 'test_app'
  user_id = 'test_user'
  
  # Create two sessions
  session1 = await ttl_session_service.create_session(
      app_name=app_name,
      user_id=user_id
  )
  session2 = await ttl_session_service.create_session(
      app_name=app_name,
      user_id=user_id
  )
  
  # Both sessions should be listed
  response = await ttl_session_service.list_sessions(
      app_name=app_name,
      user_id=user_id
  )
  assert len(response.sessions) == 2
  
  # Mock one session as expired
  def mock_is_expired(update_time):
    # Return True for session1's update time only
    return update_time == session1.last_update_time
  
  with mock.patch.object(
      ttl_session_service,
      '_is_session_expired',
      side_effect=lambda dt: mock_is_expired(dt.timestamp())
  ):
    # Only session2 should be listed
    response = await ttl_session_service.list_sessions(
        app_name=app_name,
        user_id=user_id
    )
    assert len(response.sessions) == 1
    assert response.sessions[0].id == session2.id


@pytest.mark.asyncio
async def test_cleanup_expired_sessions_requires_ttl(no_ttl_session_service):
  """Test that cleanup_expired_sessions raises error without TTL."""
  with pytest.raises(ValueError, match='ttl_seconds is not configured'):
    await no_ttl_session_service.cleanup_expired_sessions()


@pytest.mark.asyncio
async def test_cleanup_expired_sessions_removes_old_sessions(
    ttl_session_service
):
  """Test that cleanup_expired_sessions physically deletes expired sessions."""
  app_name = 'test_app'
  user_id = 'test_user'
  
  # Create a session
  session = await ttl_session_service.create_session(
      app_name=app_name,
      user_id=user_id
  )
  session_id = session.id
  
  # Manually update the session's update_time in the database to be old
  schema = ttl_session_service._get_schema_classes()
  async with ttl_session_service.database_session_factory() as sql_session:
    storage_session = await sql_session.get(
        schema.StorageSession, (app_name, user_id, session_id)
    )
    old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
    # For SQLite, use naive datetime
    if storage_session._dialect_name == 'sqlite':
      old_time = old_time.replace(tzinfo=None)
    storage_session.update_time = old_time
    await sql_session.commit()
  
  # Verify session is now considered expired
  retrieved = await ttl_session_service.get_session(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id
  )
  assert retrieved is None
  
  # Run cleanup
  deleted_count = await ttl_session_service.cleanup_expired_sessions()
  assert deleted_count == 1
  
  # Verify session is physically deleted from database
  async with ttl_session_service.database_session_factory() as sql_session:
    storage_session = await sql_session.get(
        schema.StorageSession, (app_name, user_id, session_id)
    )
    assert storage_session is None


@pytest.mark.asyncio
async def test_cleanup_expired_sessions_preserves_active_sessions(
    ttl_session_service
):
  """Test that cleanup_expired_sessions keeps non-expired sessions."""
  app_name = 'test_app'
  user_id = 'test_user'
  
  # Create two sessions
  old_session = await ttl_session_service.create_session(
      app_name=app_name,
      user_id=user_id
  )
  new_session = await ttl_session_service.create_session(
      app_name=app_name,
      user_id=user_id
  )
  
  # Make the first session old
  schema = ttl_session_service._get_schema_classes()
  async with ttl_session_service.database_session_factory() as sql_session:
    storage_session = await sql_session.get(
        schema.StorageSession, (app_name, user_id, old_session.id)
    )
    old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
    # For SQLite, use naive datetime
    if storage_session._dialect_name == 'sqlite':
      old_time = old_time.replace(tzinfo=None)
    storage_session.update_time = old_time
    await sql_session.commit()
  
  # Run cleanup
  deleted_count = await ttl_session_service.cleanup_expired_sessions()
  assert deleted_count == 1
  
  # Verify new session still exists
  retrieved = await ttl_session_service.get_session(
      app_name=app_name,
      user_id=user_id,
      session_id=new_session.id
  )
  assert retrieved is not None
  assert retrieved.id == new_session.id
  
  # Verify old session is gone
  retrieved_old = await ttl_session_service.get_session(
      app_name=app_name,
      user_id=user_id,
      session_id=old_session.id
  )
  assert retrieved_old is None


@pytest.mark.asyncio
async def test_is_session_expired_with_ttl():
  """Test _is_session_expired method with TTL configured."""
  service = DatabaseSessionService(
      'sqlite+aiosqlite:///:memory:',
      ttl_seconds=60
  )
  
  now = datetime.now(timezone.utc)
  
  # Recent session should not be expired
  recent_time = now - timedelta(seconds=30)
  assert not service._is_session_expired(recent_time)
  
  # Old session should be expired
  old_time = now - timedelta(seconds=120)
  assert service._is_session_expired(old_time)
  
  # Edge case: exactly at TTL
  edge_time = now - timedelta(seconds=60)
  # Should not be expired (age == ttl, not > ttl)
  assert not service._is_session_expired(edge_time)
  
  await service.close()


@pytest.mark.asyncio
async def test_is_session_expired_without_ttl():
  """Test _is_session_expired returns False when TTL is not configured."""
  service = DatabaseSessionService('sqlite+aiosqlite:///:memory:')
  
  now = datetime.now(timezone.utc)
  old_time = now - timedelta(days=365)
  
  # Without TTL, sessions never expire
  assert not service._is_session_expired(old_time)
  assert not service._is_session_expired(now)
  
  await service.close()


@pytest.mark.asyncio
async def test_is_session_expired_handles_naive_datetime():
  """Test _is_session_expired handles naive datetime objects."""
  service = DatabaseSessionService(
      'sqlite+aiosqlite:///:memory:',
      ttl_seconds=60
  )
  
  # Create a naive datetime (no timezone info)
  now = datetime.now()
  old_time = now - timedelta(seconds=120)
  
  # Should handle naive datetime by assuming UTC
  assert service._is_session_expired(old_time)
  
  await service.close()


@pytest.mark.asyncio
async def test_cleanup_expired_sessions_returns_zero_when_none_expired(
    ttl_session_service
):
  """Test that cleanup returns 0 when no sessions are expired."""
  app_name = 'test_app'
  user_id = 'test_user'
  
  # Create a fresh session
  await ttl_session_service.create_session(
      app_name=app_name,
      user_id=user_id
  )
  
  # Run cleanup - should delete nothing
  deleted_count = await ttl_session_service.cleanup_expired_sessions()
  assert deleted_count == 0


@pytest.mark.asyncio
async def test_ttl_with_multiple_apps_and_users(ttl_session_service):
  """Test TTL works correctly across multiple apps and users."""
  app1, app2 = 'app1', 'app2'
  user1, user2 = 'user1', 'user2'
  
  # Create sessions for different apps and users
  s1 = await ttl_session_service.create_session(app_name=app1, user_id=user1)
  s2 = await ttl_session_service.create_session(app_name=app1, user_id=user2)
  s3 = await ttl_session_service.create_session(app_name=app2, user_id=user1)
  s4 = await ttl_session_service.create_session(app_name=app2, user_id=user2)
  
  # Make s1 and s3 old
  schema = ttl_session_service._get_schema_classes()
  async with ttl_session_service.database_session_factory() as sql_session:
    for session_id, app, user in [
        (s1.id, app1, user1),
        (s3.id, app2, user1),
    ]:
      storage_session = await sql_session.get(
          schema.StorageSession, (app, user, session_id)
      )
      old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
      if storage_session._dialect_name == 'sqlite':
        old_time = old_time.replace(tzinfo=None)
      storage_session.update_time = old_time
    await sql_session.commit()
  
  # Cleanup should remove 2 sessions
  deleted_count = await ttl_session_service.cleanup_expired_sessions()
  assert deleted_count == 2
  
  # Verify only s2 and s4 remain
  assert await ttl_session_service.get_session(
      app_name=app1, user_id=user1, session_id=s1.id
  ) is None
  assert await ttl_session_service.get_session(
      app_name=app1, user_id=user2, session_id=s2.id
  ) is not None
  assert await ttl_session_service.get_session(
      app_name=app2, user_id=user1, session_id=s3.id
  ) is None
  assert await ttl_session_service.get_session(
      app_name=app2, user_id=user2, session_id=s4.id
  ) is not None
