# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
# with the License. A copy of the License is located at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# or in the 'license' file accompanying this file. This file is distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, express or implied. See the License for the specific language governing permissions
# and limitations under the License.
"""Tests for Git Repository Research MCP Server with a local repository."""

import pytest

# Import the server functionality
from awslabs.git_repo_research_mcp_server.repository import clone_repository, is_git_repo


@pytest.mark.asyncio
async def test_repository_indexing():
    """Test various errors for repository."""
    try:
        assert not is_git_repo('not-a-real-repo')

        with pytest.raises(Exception):
            clone_repository(url='not-a-real-url')

    except Exception as e:
        assert 'Error testing repository' in str(e)
