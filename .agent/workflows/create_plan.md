---
description: Create a new implementation plan using AgentQMS
---
1. Determine the appropriate NAME (slug) and TITLE for the plan.
2. Run the creation command:
   ```bash
   cd AgentQMS/interface && make create-plan NAME=plan-slug TITLE="Plan Title"
   ```
   *Example: `cd AgentQMS/interface && make create-plan NAME=auth-refactor TITLE="Authentication System Refactor"`*
3. The plan will be created in `docs/artifacts/`. Use `view_file` to read it and fill in the details.
