#!/bin/bash

# Simple bash script to process seismic data day by day using cut_waveforms_single_day.py
# This approach saves memory by processing one day at a time
# Modified to support multi-month processing with START_DATE and END_DATE

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Time period parameters (YYYY-MM-DD format)
START_DATE="2025-01-01"
END_DATE="2025-09-30"  # Adjust date range as needed

# Directory parameters
BASEDIR="/Volumes/GeoPhysics_49/users-data/montalca"
DATADIR="/Volumes/GeoPhysics_49/users-data/montalca/DATA"
PARALLEL_JOBS=6  # Number of days to process in parallel

# Python environment
PYTHON_SCRIPT="/Volumes/GeoPhysics_49/users-data/montalca/PROGRAMS/PYTHON/EARTHQUAKE_CATALOGUE/cut_waveforms_single_day.py"
CONDA_ENV="seisbench"

# Log directory
LOG_DIR="/Volumes/GeoPhysics_49/users-data/montalca/PROGRAMS/PYTHON/OUT_LOGS"
mkdir -p "$LOG_DIR"

# =============================================================================
# COLORS FOR OUTPUT
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Function to generate julian days for date range
generate_julian_days() {
    local start_date="$1"
    local end_date="$2"
    
    python3 -c "
from datetime import datetime, timedelta

try:
    start = datetime.strptime('$start_date', '%Y-%m-%d')
    end = datetime.strptime('$end_date', '%Y-%m-%d')
    current = start
    
    while current <= end:
        julian_day = current.strftime('%j')
        year = current.strftime('%Y')
        date_str = current.strftime('%Y-%m-%d')
        print(f'{year},{julian_day},{date_str}')
        current += timedelta(days=1)
except Exception as e:
    print(f'Error: {e}')
    exit(1)
"
}

# Function to display configuration
display_config() {
    echo "=================================================="
    echo -e "${PURPLE}CUT WAVEFORMS CONFIGURATION${NC}"
    echo "=================================================="
    echo -e "Start date: ${YELLOW}$START_DATE${NC}"
    echo -e "End date: ${YELLOW}$END_DATE${NC}"
    echo -e "Base directory: ${YELLOW}$BASEDIR${NC}"
    echo -e "Data directory: ${YELLOW}$DATADIR${NC}"
    echo -e "Conda environment: ${YELLOW}$CONDA_ENV${NC}"
    echo -e "Parallel jobs: ${YELLOW}$PARALLEL_JOBS${NC}"
    echo -e "Python script: ${YELLOW}$PYTHON_SCRIPT${NC}"
    echo -e "Log directory: ${YELLOW}$LOG_DIR${NC}"
    echo "=================================================="
}

# =============================================================================
# CONFIGURATION DISPLAY
# =============================================================================

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}    CUT WAVEFORMS - MULTI-MONTH PROCESSING${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

display_config
echo ""

# =============================================================================
# VALIDATION
# =============================================================================

echo -e "${YELLOW}Validating configuration...${NC}"

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Check if environment was activated successfully
if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]; then
    echo -e "${RED}Error: Failed to activate conda environment $CONDA_ENV${NC}"
    echo "Available environments:"
    conda env list
    exit 1
fi

echo -e "${GREEN}✓ Conda environment $CONDA_ENV activated successfully${NC}"
echo ""

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: Python script $PYTHON_SCRIPT not found!${NC}"
    exit 1
fi

# Check if directories exist
if [ ! -d "$BASEDIR" ]; then
    echo -e "${RED}Error: Base directory $BASEDIR does not exist!${NC}"
    exit 1
fi

if [ ! -d "$DATADIR" ]; then
    echo -e "${RED}Error: Data directory $DATADIR does not exist!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Configuration validated successfully${NC}"
echo ""

# =============================================================================
# GENERATE DAY LIST
# =============================================================================

echo -e "${YELLOW}Generating list of days to process...${NC}"

# Generate list of days to process
JULIAN_DAYS=$(generate_julian_days "$START_DATE" "$END_DATE")
if [ -z "$JULIAN_DAYS" ]; then
    echo -e "${RED}Error: Failed to generate julian days for date range${NC}"
    exit 1
fi

# Count total days
TOTAL_DAYS=$(echo "$JULIAN_DAYS" | wc -l | tr -d ' ')
echo -e "${GREEN}✓ Found $TOTAL_DAYS days to process${NC}"
echo ""

# Statistics
SUCCESSFUL_DAYS=0
FAILED_DAYS=0
FAILED_DAYS_LIST=()

# Start timing
SCRIPT_START_TIME=$(date +%s)
echo "Processing started at: $(date)"
echo ""

# =============================================================================
# PROCESSING FUNCTION
# =============================================================================

# Function to process a single day
process_day() {
    local year=$1
    local jday=$2
    local date_str=$3
    local DAY_START_TIME=$(date +%s)
    
    echo -e "${YELLOW}[PID $$] Processing $date_str (Year: $year, JDay: $jday)${NC}"
    
    # Run the Python script for this day
    python "$PYTHON_SCRIPT" \
        --year "$year" \
        --jday "$jday" \
        --basedir "$BASEDIR" \
        --datadir "$DATADIR" 2>&1
    
    local exit_code=$?
    local DAY_END_TIME=$(date +%s)
    local DAY_DURATION=$((DAY_END_TIME - DAY_START_TIME))
    local DAY_MINUTES=$((DAY_DURATION / 60))
    local DAY_SECONDS=$((DAY_DURATION % 60))
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ [PID $$] $date_str (JDay $jday) completed successfully in ${DAY_MINUTES}m ${DAY_SECONDS}s${NC}"
        echo "SUCCESS:$year:$jday:$date_str:$DAY_DURATION" >> /tmp/processing_results_$$.tmp
    else
        echo -e "${RED}✗ [PID $$] $date_str (JDay $jday) failed after ${DAY_MINUTES}m ${DAY_SECONDS}s${NC}"
        echo "FAILED:$year:$jday:$date_str:$DAY_DURATION" >> /tmp/processing_results_$$.tmp
    fi
}

# Export function and variables for parallel execution
export -f process_day
export PYTHON_SCRIPT BASEDIR DATADIR RED GREEN YELLOW NC

# =============================================================================
# PARALLEL PROCESSING
# =============================================================================

# Process days in parallel batches
echo "Processing $TOTAL_DAYS days in batches of $PARALLEL_JOBS..."
echo ""

# Clean up any previous temp files
rm -f /tmp/processing_results_*.tmp

# Convert julian days to array for batch processing
mapfile -t DAY_ARRAY <<< "$JULIAN_DAYS"

# Process days in parallel batches
batch_count=0
for ((i=0; i<${#DAY_ARRAY[@]}; i+=PARALLEL_JOBS)); do
    batch_count=$((batch_count + 1))
    batch_end=$((i + PARALLEL_JOBS - 1))
    if [ $batch_end -ge ${#DAY_ARRAY[@]} ]; then
        batch_end=$((${#DAY_ARRAY[@]} - 1))
    fi
    
    # Get date range for this batch
    first_date=$(echo "${DAY_ARRAY[$i]}" | cut -d',' -f3)
    last_date=$(echo "${DAY_ARRAY[$batch_end]}" | cut -d',' -f3)
    
    echo -e "${CYAN}=== Batch $batch_count: $first_date to $last_date ===${NC}"
    
    # Start parallel processes for this batch
    for ((j=i; j<=batch_end && j<${#DAY_ARRAY[@]}; j++)); do
        IFS=',' read -r year jday date_str <<< "${DAY_ARRAY[$j]}"
        process_day "$year" "$jday" "$date_str" &
    done
    
    # Wait for all processes in this batch to complete
    wait
    
    echo -e "${CYAN}=== Batch $batch_count completed ===${NC}"
    echo ""
    
    # Small delay between batches to avoid overwhelming the system
    sleep 2
done

# =============================================================================
# RESULTS COLLECTION
# =============================================================================

# Calculate total processing time
SCRIPT_END_TIME=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

# Collect results from temporary files
SUCCESSFUL_DAYS=0
FAILED_DAYS=0
FAILED_DAYS_LIST=()
TOTAL_PROCESSING_TIME=0

for result_file in /tmp/processing_results_*.tmp; do
    if [ -f "$result_file" ]; then
        while IFS=':' read -r status year jday date_str duration; do
            if [ "$status" = "SUCCESS" ]; then
                ((SUCCESSFUL_DAYS++))
                TOTAL_PROCESSING_TIME=$((TOTAL_PROCESSING_TIME + duration))
            elif [ "$status" = "FAILED" ]; then
                ((FAILED_DAYS++))
                FAILED_DAYS_LIST+=("$year-$jday ($date_str)")
                TOTAL_PROCESSING_TIME=$((TOTAL_PROCESSING_TIME + duration))
            fi
        done < "$result_file"
    fi
done

# Clean up temporary files
rm -f /tmp/processing_results_*.tmp

# =============================================================================
# FINAL SUMMARY
# =============================================================================

echo "=================================================="
echo -e "${PURPLE}PROCESSING COMPLETE${NC}"
echo "=================================================="
echo "Processing finished at: $(date)"
echo "Date range: $START_DATE to $END_DATE"
echo "Total wall-clock time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "Total days processed: $TOTAL_DAYS"
echo -e "Successful days: ${GREEN}$SUCCESSFUL_DAYS${NC}"
echo -e "Failed days: ${RED}$FAILED_DAYS${NC}"

# Calculate averages if we have successful days
if [ $SUCCESSFUL_DAYS -gt 0 ]; then
    AVG_TIME_PER_DAY=$((TOTAL_PROCESSING_TIME / SUCCESSFUL_DAYS))
    AVG_MINUTES=$((AVG_TIME_PER_DAY / 60))
    AVG_SECONDS=$((AVG_TIME_PER_DAY % 60))
    echo "Average processing time per successful day: ${AVG_MINUTES}m ${AVG_SECONDS}s"
    
    # # Calculate speedup achieved by parallel processing
    # ESTIMATED_SEQUENTIAL_TIME=$((AVG_TIME_PER_DAY * SUCCESSFUL_DAYS))
    # EST_SEQ_HOURS=$((ESTIMATED_SEQUENTIAL_TIME / 3600))
    # EST_SEQ_MINUTES=$(((ESTIMATED_SEQUENTIAL_TIME % 3600) / 60))
    # SPEEDUP=$(echo "scale=1; $ESTIMATED_SEQUENTIAL_TIME / $TOTAL_DURATION" | bc -l 2>/dev/null || echo "N/A")
    
    # echo "Estimated sequential time would have been: ${EST_SEQ_HOURS}h ${EST_SEQ_MINUTES}m"
    # if [ "$SPEEDUP" != "N/A" ]; then
    #     echo "Speedup achieved: ${SPEEDUP}x"
    # fi
    
    # # Estimate time for remaining processing
    # if [ $SUCCESSFUL_DAYS -ge 3 ]; then
    #     # Estimate for 6 months (182 days)
    #     DAYS_PER_BATCH=$PARALLEL_JOBS
    #     BATCHES_FOR_6MONTHS=$(((182 + DAYS_PER_BATCH - 1) / DAYS_PER_BATCH))
    #     CURRENT_BATCHES=$(((TOTAL_DAYS + PARALLEL_JOBS - 1) / PARALLEL_JOBS))
    #     ESTIMATED_6MONTH_TIME=$((TOTAL_DURATION * BATCHES_FOR_6MONTHS / CURRENT_BATCHES))
    #     EST_HOURS=$((ESTIMATED_6MONTH_TIME / 3600))
    #     EST_MINUTES=$(((ESTIMATED_6MONTH_TIME % 3600) / 60))
    #     echo "Estimated wall-clock time for 6 months (182 days): ${EST_HOURS}h ${EST_MINUTES}m"
    # fi
fi

if [ $FAILED_DAYS -gt 0 ]; then
    echo -e "${RED}Failed days list:${NC}"
    for failed_day in "${FAILED_DAYS_LIST[@]}"; do
        echo "  - $failed_day"
    done
    echo ""
    echo "You can rerun failed days individually by adjusting START_DATE and END_DATE"
else
    echo -e "${GREEN}All days processed successfully!${NC}"
fi

# Calculate success rate
SUCCESS_RATE=$((SUCCESSFUL_DAYS * 100 / TOTAL_DAYS))
echo "Success rate: $SUCCESS_RATE%"
echo "=================================================="

# Exit with appropriate code
if [ $FAILED_DAYS -eq 0 ]; then
    exit 0
else
    exit 1
fi
