"""
Orchestrator Agent

Coordinates multi-agent OCR workflows and task distribution.

Responsibilities:
- Receive high-level OCR tasks
- Decompose tasks into subtasks
- Delegate to specialized agents
- Aggregate results
- Handle error recovery
"""

import logging
from typing import Any, Optional, Dict, List
import json
import time
import uuid

from ocr.agents.base_agent import LLMAgent, AgentCapability

logger = logging.getLogger(__name__)


class OrchestratorAgent(LLMAgent):
    """
    Orchestrates multi-agent OCR workflows.

    Acts as the central coordinator for complex OCR pipelines,
    delegating tasks to specialized agents and aggregating results.
    """

    def __init__(
        self,
        agent_id: str = "agent.orchestrator",
        rabbitmq_host: str = "rabbitmq",
        llm_provider: str = "grok4"  # Use Grok4 for complex reasoning
    ):
        """Initialize orchestrator agent."""
        capabilities = [
            AgentCapability(
                name="execute_ocr_workflow",
                description="Execute complete OCR workflow on image(s)",
                input_schema={
                    "image_paths": "list[str] or str",
                    "workflow_config": "dict (optional)",
                    "output_dir": "str (optional)"
                },
                output_schema={
                    "status": "str",
                    "workflow_id": "str",
                    "results": "dict",
                    "summary": "dict"
                }
            ),
            AgentCapability(
                name="plan_workflow",
                description="Plan OCR workflow based on requirements using LLM",
                input_schema={
                    "requirements": "str",
                    "constraints": "dict (optional)"
                },
                output_schema={
                    "status": "str",
                    "workflow_plan": "dict",
                    "estimated_time": "float"
                }
            ),
            AgentCapability(
                name="monitor_workflow",
                description="Monitor status of running workflow",
                input_schema={
                    "workflow_id": "str"
                },
                output_schema={
                    "status": "str",
                    "workflow_status": "str",
                    "progress": "float",
                    "current_stage": "str"
                }
            )
        ]

        super().__init__(
            agent_id=agent_id,
            agent_type="orchestrator",
            llm_provider=llm_provider,
            rabbitmq_host=rabbitmq_host,
            capabilities=capabilities
        )

        # Register handlers
        self.register_handler("cmd.execute_ocr_workflow", self._handle_execute_workflow)
        self.register_handler("cmd.plan_workflow", self._handle_plan_workflow)
        self.register_handler("cmd.monitor_workflow", self._handle_monitor_workflow)

        # Track active workflows
        self._active_workflows: Dict[str, dict] = {}

    def get_binding_keys(self) -> list[str]:
        """Return routing keys for this agent."""
        return [
            "cmd.execute_ocr_workflow.#",
            "cmd.plan_workflow.#",
            "cmd.monitor_workflow.#"
        ]

    def _handle_execute_workflow(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Handle OCR workflow execution request."""
        payload = envelope.get("payload", {})
        image_paths = payload.get("image_paths")
        workflow_config = payload.get("workflow_config", {})
        output_dir = payload.get("output_dir")

        if not image_paths:
            return {"status": "error", "message": "image_paths is required"}

        # Normalize to list
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Generate workflow ID
        workflow_id = str(uuid.uuid4())

        try:
            logger.info(f"Starting workflow {workflow_id} for {len(image_paths)} images")

            # Track workflow
            self._active_workflows[workflow_id] = {
                "status": "running",
                "started_at": time.time(),
                "image_count": len(image_paths),
                "current_stage": "initialization"
            }

            # Default workflow stages
            stages = workflow_config.get("stages", [
                "preprocess",
                "inference",
                "validation"
            ])

            all_results = []

            for img_path in image_paths:
                logger.info(f"Processing {img_path}")
                image_result = {"image": img_path, "stages": {}}

                # Stage 1: Preprocessing (if enabled)
                if "preprocess" in stages:
                    self._active_workflows[workflow_id]["current_stage"] = "preprocessing"
                    preprocess_result = self._execute_preprocessing(
                        img_path,
                        workflow_config.get("preprocess_options", {})
                    )
                    image_result["stages"]["preprocess"] = preprocess_result

                    # Use preprocessed image for next stages
                    if preprocess_result.get("status") == "success":
                        processed_img = preprocess_result.get("output_path", img_path)
                    else:
                        processed_img = img_path
                else:
                    processed_img = img_path

                # Stage 2: OCR Inference
                if "inference" in stages:
                    self._active_workflows[workflow_id]["current_stage"] = "inference"
                    inference_result = self._execute_inference(
                        processed_img,
                        workflow_config.get("inference_options", {})
                    )
                    image_result["stages"]["inference"] = inference_result

                # Stage 3: Validation (if enabled)
                if "validation" in stages and "inference" in image_result["stages"]:
                    self._active_workflows[workflow_id]["current_stage"] = "validation"
                    validation_result = self._execute_validation(
                        image_result["stages"]["inference"],
                        workflow_config.get("validation_options", {})
                    )
                    image_result["stages"]["validation"] = validation_result

                all_results.append(image_result)

            # Update workflow status
            self._active_workflows[workflow_id]["status"] = "completed"
            self._active_workflows[workflow_id]["current_stage"] = "finished"
            self._active_workflows[workflow_id]["completed_at"] = time.time()

            # Calculate summary
            summary = self._calculate_workflow_summary(all_results, workflow_id)

            # Save results if output_dir specified
            if output_dir:
                from pathlib import Path
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                result_file = output_path / f"workflow_{workflow_id}_results.json"
                with open(result_file, "w") as f:
                    json.dump({
                        "workflow_id": workflow_id,
                        "results": all_results,
                        "summary": summary
                    }, f, indent=2)

            return {
                "status": "success",
                "workflow_id": workflow_id,
                "results": all_results,
                "summary": summary
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            self._active_workflows[workflow_id]["status"] = "failed"
            self._active_workflows[workflow_id]["error"] = str(e)
            return {"status": "error", "message": str(e), "workflow_id": workflow_id}

    def _execute_preprocessing(self, image_path: str, options: dict) -> dict[str, Any]:
        """Execute preprocessing stage."""
        try:
            result = self.send_command(
                target="agent.ocr.preprocessor",
                command="normalize_image",
                payload={
                    "image_path": image_path,
                    "options": options
                },
                timeout=30
            )
            return result.get("payload", {})
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return {"status": "error", "message": str(e)}

    def _execute_inference(self, image_path: str, options: dict) -> dict[str, Any]:
        """Execute OCR inference stage."""
        try:
            result = self.send_command(
                target="agent.ocr.inference",
                command="full_ocr",
                payload={
                    "image_path": image_path,
                    **options
                },
                timeout=60
            )
            return result.get("payload", {})
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {"status": "error", "message": str(e)}

    def _execute_validation(self, inference_result: dict, options: dict) -> dict[str, Any]:
        """Execute validation stage."""
        try:
            result = self.send_command(
                target="agent.ocr.validator",
                command="validate_ocr_result",
                payload={
                    "ocr_result": inference_result,
                    "validation_rules": options,
                    "use_llm": options.get("use_llm", True)
                },
                timeout=45
            )
            return result.get("payload", {})
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"status": "error", "message": str(e)}

    def _calculate_workflow_summary(self, results: List[dict], workflow_id: str) -> dict[str, Any]:
        """Calculate workflow execution summary."""
        workflow_info = self._active_workflows.get(workflow_id, {})

        total_images = len(results)
        successful_images = sum(
            1 for r in results
            if r.get("stages", {}).get("inference", {}).get("status") == "success"
        )

        total_time = workflow_info.get("completed_at", time.time()) - workflow_info.get("started_at", 0)

        # Calculate average quality score if validation was performed
        quality_scores = []
        for result in results:
            validation = result.get("stages", {}).get("validation", {})
            if validation.get("quality_score"):
                quality_scores.append(validation["quality_score"])

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None

        return {
            "workflow_id": workflow_id,
            "total_images": total_images,
            "successful": successful_images,
            "failed": total_images - successful_images,
            "total_time_seconds": total_time,
            "avg_time_per_image": total_time / total_images if total_images > 0 else 0,
            "avg_quality_score": avg_quality
        }

    def _handle_plan_workflow(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Handle workflow planning request using LLM."""
        payload = envelope.get("payload", {})
        requirements = payload.get("requirements")
        constraints = payload.get("constraints", {})

        if not requirements:
            return {"status": "error", "message": "requirements is required"}

        try:
            system_prompt = """You are an OCR workflow planning expert.
            Given user requirements, design an optimal OCR workflow plan.

            Available stages:
            - preprocess: Image preprocessing (normalization, enhancement, background removal)
            - inference: OCR text detection and recognition
            - validation: Quality validation and error detection

            Available options for each stage can be configured.

            Return a JSON workflow plan."""

            prompt = f"""Design an OCR workflow plan for:

Requirements: {requirements}

Constraints: {json.dumps(constraints, indent=2)}

Return JSON format:
{{
    "stages": ["stage1", "stage2", ...],
    "preprocess_options": {{}},
    "inference_options": {{}},
    "validation_options": {{}},
    "estimated_time_per_image": float,
    "reasoning": "explanation of the plan"
}}"""

            response = self.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                max_tokens=1500
            )

            # Parse LLM response
            try:
                workflow_plan = json.loads(response)
            except json.JSONDecodeError:
                workflow_plan = {
                    "stages": ["preprocess", "inference", "validation"],
                    "reasoning": response
                }

            return {
                "status": "success",
                "workflow_plan": workflow_plan,
                "estimated_time": workflow_plan.get("estimated_time_per_image", 5.0)
            }

        except Exception as e:
            logger.error(f"Workflow planning failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _handle_monitor_workflow(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Handle workflow monitoring request."""
        payload = envelope.get("payload", {})
        workflow_id = payload.get("workflow_id")

        if not workflow_id:
            return {"status": "error", "message": "workflow_id is required"}

        if workflow_id not in self._active_workflows:
            return {"status": "error", "message": f"Workflow not found: {workflow_id}"}

        workflow = self._active_workflows[workflow_id]

        # Calculate progress
        elapsed = time.time() - workflow.get("started_at", time.time())

        return {
            "status": "success",
            "workflow_id": workflow_id,
            "workflow_status": workflow.get("status"),
            "current_stage": workflow.get("current_stage"),
            "elapsed_time": elapsed,
            "image_count": workflow.get("image_count"),
            "error": workflow.get("error")
        }


if __name__ == "__main__":
    # Run the agent
    agent = OrchestratorAgent()
    agent.start()
