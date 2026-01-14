# Task list

- [/] MCP Deployment & Optimization <!-- id: 0 -->
    - [x] Fix Server 500/Crash Issues <!-- id: 1 -->
        - [x] Adapt `unified_server.py` for Cloud Environment <!-- id: 2 -->
        - [x] Update `.devcontainer.json` <!-- id: 3 -->
        - [x] Verify server startup <!-- id: 4 -->
    - [x] Artifact Pruning <!-- id: 5 -->
        - [x] Create `archive_artifacts.sh` <!-- id: 6 -->
        - [x] Prune items >30 days old <!-- id: 7 -->
        - [x] Perform manual archive of walkthroughs/bugs (User Request) <!-- id: 8 -->
    - [x] Proactive Feedback <!-- id: 9 -->
        - [x] Implement `ProactiveFeedbackInterceptor` <!-- id: 10 -->
        - [x] Integrate into `unified_server.py` <!-- id: 11 -->
    - [x] Context Auto-Loading <!-- id: 12 -->
        - [x] Implement `auto_suggest_context` logic in server <!-- id: 13 -->
    - [x] Silent Failure Detection <!-- id: 14 -->
        - [x] Implement `HealthMonitor` class <!-- id: 15 -->
    - [/] Observability Dashboard (Next) <!-- id: 16 -->
