#!/usr/bin/env bash
"""
VLM validation for sepia enhancement results.

Uses Dashscope Qwen3 VL Plus to validate sepia enhancement quality
compared to other methods (raw, gray-scale, gray-world normalization).

EDS v1.0 Experiment: 20251217_024343_image_enhancements_implementation
Phase: Week 2 Day 5
Target: Visual validation of sepia superiority hypothesis

Usage:
    # Validate single comparison grid
    ./vlm_validate_sepia.sh <comparison_grid_image>

    # Validate all sepia results in directory
    ./vlm_validate_sepia.sh <output_dir>
"""

set -euo pipefail

# Configuration
VLM_MODEL="qwen-vl-plus"
API_KEY="${DASHSCOPE_API_KEY:-}"

# Prompt for sepia enhancement validation
SEPIA_VALIDATION_PROMPT="You are evaluating document image enhancement methods for OCR preprocessing.

Compare the following enhancement methods shown in this image:
1. Raw - Original unprocessed image
2. Grayscale - Simple grayscale conversion
3. Gray World Norm - Gray-world background normalization
4. Sepia Classic - Traditional sepia tone
5. Sepia Adaptive - Adaptive sepia with contrast preservation
6. Sepia Warm - Enhanced warm tones for OCR
7. Sepia Contrast - Sepia with CLAHE enhancement

For each method, evaluate:
1. **Text Clarity** (1-10): How clearly is text visible and readable?
2. **Background Quality** (1-10): How clean and uniform is the background?
3. **Color Tint Impact** (1-10): How well does it handle color tints/variations?
4. **OCR Suitability** (1-10): How suitable for OCR processing?

Provide:
- Individual scores for each method
- Ranking from best to worst for OCR
- Key observations about sepia methods vs alternatives
- Recommendation for pipeline integration

Format your response as structured JSON."

# Check dependencies
if [ -z "$API_KEY" ]; then
    echo "Error: DASHSCOPE_API_KEY environment variable not set"
    echo "Please set it with: export DASHSCOPE_API_KEY='your_key_here'"
    exit 1
fi

command -v jq >/dev/null 2>&1 || {
    echo "Error: jq is required but not installed"
    exit 1
}

command -v curl >/dev/null 2>&1 || {
    echo "Error: curl is required but not installed"
    exit 1
}

# Function to validate a single image
validate_image() {
    local image_path="$1"
    local output_path="$2"

    echo "Validating: $image_path"

    # Convert image to base64
    local base64_image
    base64_image=$(base64 -w 0 "$image_path")

    # Prepare API request
    local request_body
    request_body=$(jq -n \
        --arg model "$VLM_MODEL" \
        --arg prompt "$SEPIA_VALIDATION_PROMPT" \
        --arg image "data:image/jpeg;base64,$base64_image" \
        '{
            model: $model,
            input: {
                messages: [
                    {
                        role: "user",
                        content: [
                            {type: "text", text: $prompt},
                            {type: "image_url", image_url: {url: $image}}
                        ]
                    }
                ]
            },
            parameters: {
                result_format: "message"
            }
        }')

    # Call API
    local response
    response=$(curl -s -X POST "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "$request_body")

    # Check for errors
    if echo "$response" | jq -e '.code' >/dev/null 2>&1; then
        echo "Error from API:"
        echo "$response" | jq -r '.message'
        return 1
    fi

    # Extract and save response
    echo "$response" | jq '.' > "$output_path"

    # Extract text content for display
    local content
    content=$(echo "$response" | jq -r '.output.choices[0].message.content // empty')

    if [ -n "$content" ]; then
        echo "VLM Response saved to: $output_path"
        echo ""
        echo "=== VLM Assessment ==="
        echo "$content"
        echo "====================="
    else
        echo "Warning: No content in response"
    fi

    return 0
}

# Main execution
main() {
    local input="$1"

    if [ ! -e "$input" ]; then
        echo "Error: Input path does not exist: $input"
        exit 1
    fi

    if [ -f "$input" ]; then
        # Single file
        local output_dir
        output_dir=$(dirname "$input")
        local output_file
        output_file="$output_dir/$(basename "$input" .jpg)_vlm_validation.json"

        validate_image "$input" "$output_file"

    elif [ -d "$input" ]; then
        # Directory - find all comparison grids
        local comparison_files
        mapfile -t comparison_files < <(find "$input" -name "*_comparison_grid.jpg" -type f)

        if [ ${#comparison_files[@]} -eq 0 ]; then
            echo "No comparison grid files found in $input"
            echo "Looking for files matching: *_comparison_grid.jpg"
            exit 1
        fi

        echo "Found ${#comparison_files[@]} comparison grid(s)"
        echo ""

        for file in "${comparison_files[@]}"; do
            local output_file="${file%.jpg}_vlm_validation.json"
            validate_image "$file" "$output_file"
            echo ""
            echo "---"
            echo ""
        done

        echo "✓ Validation complete for ${#comparison_files[@]} file(s)"
    else
        echo "Error: Input must be a file or directory"
        exit 1
    fi
}

# Run main if executed directly
if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
    if [ $# -lt 1 ]; then
        echo "Usage: $0 <image_file_or_directory>"
        echo ""
        echo "Examples:"
        echo "  $0 outputs/comparison_grid.jpg"
        echo "  $0 outputs/sepia_tests/"
        exit 1
    fi

    main "$@"
fi
