"""
OCR Validation Agent

Handles OCR result validation and quality assurance including:
- Text validation against schemas
- Quality scoring
- Error detection
- LLM-based validation
"""

import logging
from typing import Any, Optional
import json

from ocr.agents.base_agent import LLMAgent, AgentCapability

logger = logging.getLogger(__name__)


class OCRValidationAgent(LLMAgent):
    """
    Specialized agent for OCR validation and quality assurance.

    Uses LLM for intelligent validation and error detection.

    Capabilities:
    - Validate OCR results
    - Score OCR quality
    - Detect common OCR errors
    - Suggest corrections
    """

    def __init__(
        self,
        agent_id: str = "agent.ocr.validator",
        rabbitmq_host: str = "rabbitmq",
        llm_provider: str = "qwen"
    ):
        """Initialize OCR validation agent."""
        capabilities = [
            AgentCapability(
                name="validate_ocr_result",
                description="Validate OCR results for quality and accuracy",
                input_schema={
                    "ocr_result": "dict",
                    "validation_rules": "dict (optional)",
                    "use_llm": "bool (default: False)"
                },
                output_schema={
                    "status": "str",
                    "is_valid": "bool",
                    "quality_score": "float",
                    "issues": "list[dict]",
                    "suggestions": "list[str]"
                }
            ),
            AgentCapability(
                name="score_quality",
                description="Score OCR quality based on multiple metrics",
                input_schema={
                    "ocr_result": "dict",
                    "ground_truth": "str (optional)"
                },
                output_schema={
                    "status": "str",
                    "overall_score": "float",
                    "metrics": "dict"
                }
            ),
            AgentCapability(
                name="detect_errors",
                description="Detect common OCR errors using LLM",
                input_schema={
                    "text": "str",
                    "context": "str (optional)"
                },
                output_schema={
                    "status": "str",
                    "errors_found": "list[dict]",
                    "corrected_text": "str (optional)"
                }
            )
        ]

        super().__init__(
            agent_id=agent_id,
            agent_type="ocr.validator",
            llm_provider=llm_provider,
            rabbitmq_host=rabbitmq_host,
            capabilities=capabilities
        )

        # Register custom handlers
        self.register_handler("cmd.validate_ocr_result", self._handle_validate_ocr_result)
        self.register_handler("cmd.score_quality", self._handle_score_quality)
        self.register_handler("cmd.detect_errors", self._handle_detect_errors)

    def get_binding_keys(self) -> list[str]:
        """Return routing keys for this agent."""
        return [
            "cmd.validate_ocr_result.#",
            "cmd.score_quality.#",
            "cmd.detect_errors.#"
        ]

    def _handle_validate_ocr_result(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Handle OCR validation request."""
        payload = envelope.get("payload", {})
        ocr_result = payload.get("ocr_result")
        validation_rules = payload.get("validation_rules", {})
        use_llm = payload.get("use_llm", False)

        if not ocr_result:
            return {"status": "error", "message": "ocr_result is required"}

        try:
            issues = []
            quality_score = 1.0

            # Basic validation
            if not isinstance(ocr_result, dict):
                return {"status": "error", "message": "ocr_result must be a dictionary"}

            results = ocr_result.get("results", [])
            full_text = ocr_result.get("full_text", "")

            # Check if results exist
            if not results:
                issues.append({
                    "type": "no_results",
                    "severity": "warning",
                    "message": "No OCR results found"
                })
                quality_score *= 0.5

            # Validate each detection
            for i, result in enumerate(results):
                # Check confidence scores
                confidence = result.get("confidence", 0.0)
                if confidence < validation_rules.get("min_confidence", 0.5):
                    issues.append({
                        "type": "low_confidence",
                        "severity": "warning",
                        "index": i,
                        "confidence": confidence,
                        "message": f"Low confidence detection: {confidence:.2f}"
                    })
                    quality_score *= 0.9

                # Check for empty text
                text = result.get("text", "")
                if not text or text.strip() == "":
                    issues.append({
                        "type": "empty_text",
                        "severity": "error",
                        "index": i,
                        "message": "Empty text detected"
                    })
                    quality_score *= 0.8

            # LLM-based validation
            suggestions = []
            if use_llm and full_text:
                try:
                    llm_validation = self._llm_validate_text(full_text)
                    issues.extend(llm_validation.get("issues", []))
                    suggestions = llm_validation.get("suggestions", [])
                    quality_score *= llm_validation.get("score_multiplier", 1.0)
                except Exception as e:
                    logger.warning(f"LLM validation failed: {e}")

            is_valid = quality_score >= validation_rules.get("min_quality_score", 0.7)

            return {
                "status": "success",
                "is_valid": is_valid,
                "quality_score": max(0.0, min(1.0, quality_score)),
                "issues": issues,
                "suggestions": suggestions,
                "result_count": len(results)
            }

        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _llm_validate_text(self, text: str) -> dict[str, Any]:
        """Use LLM to validate text quality."""
        try:
            system_prompt = """You are an OCR validation expert.
            Analyze the provided OCR text and identify potential errors, inconsistencies, or quality issues.
            Return a JSON response with:
            - issues: list of detected issues
            - suggestions: list of improvement suggestions
            - score_multiplier: float between 0.5 and 1.0 based on quality
            """

            prompt = f"""Analyze this OCR output text:

{text}

Identify:
1. Potential OCR errors (character substitutions, missing characters)
2. Formatting issues
3. Suspicious patterns
4. Overall quality assessment

Return JSON format."""

            response = self.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=500
            )

            # Parse LLM response
            try:
                validation_result = json.loads(response)
                return validation_result
            except json.JSONDecodeError:
                # Fallback if LLM doesn't return valid JSON
                return {
                    "issues": [],
                    "suggestions": [response],
                    "score_multiplier": 1.0
                }

        except Exception as e:
            logger.error(f"LLM validation error: {e}")
            return {
                "issues": [],
                "suggestions": [],
                "score_multiplier": 1.0
            }

    def _handle_score_quality(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Handle quality scoring request."""
        payload = envelope.get("payload", {})
        ocr_result = payload.get("ocr_result")
        ground_truth = payload.get("ground_truth")

        if not ocr_result:
            return {"status": "error", "message": "ocr_result is required"}

        try:
            metrics = {}

            results = ocr_result.get("results", [])
            full_text = ocr_result.get("full_text", "")

            # Calculate confidence metrics
            if results:
                confidences = [r.get("confidence", 0.0) for r in results]
                metrics["avg_confidence"] = sum(confidences) / len(confidences)
                metrics["min_confidence"] = min(confidences)
                metrics["max_confidence"] = max(confidences)

            # Calculate text metrics
            metrics["total_detections"] = len(results)
            metrics["total_characters"] = len(full_text)
            metrics["non_empty_ratio"] = sum(1 for r in results if r.get("text", "").strip()) / max(len(results), 1)

            # If ground truth provided, calculate accuracy
            if ground_truth:
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, full_text, ground_truth).ratio()
                metrics["text_similarity"] = similarity
                metrics["character_error_rate"] = 1.0 - similarity

            # Calculate overall score
            overall_score = (
                metrics.get("avg_confidence", 0.5) * 0.4 +
                metrics.get("non_empty_ratio", 0.5) * 0.3 +
                metrics.get("text_similarity", 0.7) * 0.3
            )

            return {
                "status": "success",
                "overall_score": overall_score,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"Quality scoring failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _handle_detect_errors(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Handle error detection request using LLM."""
        payload = envelope.get("payload", {})
        text = payload.get("text")
        context = payload.get("context", "")

        if not text:
            return {"status": "error", "message": "text is required"}

        try:
            system_prompt = """You are an OCR error detection expert.
            Identify common OCR errors such as:
            - Character substitutions (0/O, 1/l/I, 5/S, etc.)
            - Missing or extra spaces
            - Garbled text
            - Incorrect word boundaries

            Return JSON with detected errors and optionally corrected text."""

            prompt = f"""Analyze this OCR text for errors:

Text: {text}

Context: {context}

Return JSON format:
{{
    "errors_found": [
        {{"type": "error_type", "position": "location", "detected": "wrong", "suggested": "correct"}}
    ],
    "corrected_text": "fully corrected text if applicable"
}}"""

            response = self.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=1000
            )

            # Parse response
            try:
                result = json.loads(response)
                return {
                    "status": "success",
                    "errors_found": result.get("errors_found", []),
                    "corrected_text": result.get("corrected_text"),
                    "original_text": text
                }
            except json.JSONDecodeError:
                return {
                    "status": "success",
                    "errors_found": [],
                    "corrected_text": None,
                    "llm_response": response
                }

        except Exception as e:
            logger.error(f"Error detection failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Run the agent
    agent = OCRValidationAgent()
    agent.start()
