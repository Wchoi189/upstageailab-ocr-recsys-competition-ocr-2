import yaml
from datetime import datetime

# Reuse core paths if possible, or define here
from project_compass.src.core import SessionManager, CompassPaths

class SprintContextWizard:
    def __init__(self):
        self.paths = CompassPaths()
        self.manager = SessionManager()

    def run(self):
        print("\n🔮 Sprint Context Wizard 🔮\n")
        print("Let's set up your work cycle.")

        # 1. Phase Selection
        phases = ["01_detection", "02_recognition", "03_kie"]
        print("\n? Which roadmap phase are you working on?")
        for i, p in enumerate(phases):
            print(f"  [{i+1}] {p}")

        while True:
            try:
                choice = input("  Selection (1-3): ")
                if not choice.strip(): continue
                phase_idx = int(choice) - 1
                if 0 <= phase_idx < len(phases):
                    phase = phases[phase_idx]
                    break
            except ValueError:
                pass

        # 2. Roadmap Loading (Mocked for now, or real reading)
        # In a real implementation, we would parse project_compass/roadmap/{phase}.yml
        # For prototype, we just ask for milestone ID.
        milestone_id = input(f"\n? Active Milestone ID for {phase} (e.g. rec-optimization): ")

        # 3. Objective
        objective = input("\n? What is your specific goal for this sprint? ")

        # 4. Generate Session ID
        timestamp = datetime.now().strftime("%Y%m%d")
        slug = f"{milestone_id}-{timestamp}"

        print(f"\nConfiguration:\n  Phase: {phase}\n  Milestone: {milestone_id}\n  Objective: {objective}\n  Session ID: {slug}")
        confirm = input("\nProceed to update current_session.yml? [y/N] ")

        if confirm.lower() != 'y':
            print("Aborted.")
            return

        # Update Session
        # We access the internal path from manager
        target_file = self.paths.current_session

        with open(target_file) as f:
            data = yaml.safe_load(f) or {}

        # Update fields
        data['session_id'] = slug
        data['status'] = 'active'
        data['started_date'] = datetime.now().strftime("%Y-%m-%d")
        data['completed_date'] = None
        data['objective'] = {
            'primary_goal': objective,
            'active_pipeline': phase,
            'success_criteria': f"Complete deliverables for {milestone_id}"
        }

        # Prototype: Add link field
        data['link'] = {
            'roadmap': f"{phase}.yml",
            'milestone_id': milestone_id
        }

        with open(target_file, 'w') as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)

        print(f"\n✨ updated {target_file}")
        print("To verify: uv run python -m project_compass.cli check-env")

if __name__ == "__main__":
    wizard = SprintContextWizard()
    wizard.run()
