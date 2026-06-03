#!/bin/bash

# EQT Detection Script - Simple parallel execution of EQT_detection_single_day.py
# This script only handles parallelization, all date logic is in the Python script

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Time period parameters (YYYY-MM-DD format)
START_DATE="2025-03-01"
END_DATE="2025-10-31"  # 3 days for parallel testing

# Directory parameters
BASEDIR="/Volumes/GeoPhysics_49/users-data/montalca"
DATADIR="/Volumes/GeoPhysics_49/users-data/montalca/DATA"

# Processing parameters
PARALLEL_JOBS=1  # Number of days to process in parallel (conservative)
MODE=1           # 1=max picks, 2=avg picks, 3=both

# Python environment
PYTHON_SCRIPT="/Volumes/GeoPhysics_49/users-data/montalca/PROGRAMS/PYTHON/EARTHQUAKE_CATALOGUE/EQT_detection_single_day.py"
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

# Function to get user input with default value
get_input() {
    local prompt="$1"
    local default="$2"
    local input
    
    echo -n -e "${CYAN}$prompt [$default]: ${NC}"
    read input
    echo "${input:-$default}"
}

# Function to validate directory
validate_directory() {
    local dir="$1"
    local name="$2"
    
    if [ ! -d "$dir" ]; then
        echo -e "${RED}Error: $name directory '$dir' does not exist!${NC}"
        return 1
    fi
    return 0
}

# Function to generate julian days for date range using Python (simple version)
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
    echo -e "${PURPLE}EQT DETECTION CONFIGURATION${NC}"
    echo "=================================================="
    echo -e "Start date: ${YELLOW}$START_DATE${NC}"
    echo -e "End date: ${YELLOW}$END_DATE${NC}"
    echo -e "Base directory: ${YELLOW}$BASEDIR${NC}"
    echo -e "Data directory: ${YELLOW}$DATADIR${NC}"
    echo -e "Mode: ${YELLOW}$MODE${NC} (1=max, 2=avg, 3=both)"
    echo -e "Parallel jobs: ${YELLOW}$PARALLEL_JOBS${NC}"
    echo -e "Python script: ${YELLOW}$PYTHON_SCRIPT${NC}"
    echo -e "Log directory: ${YELLOW}$LOG_DIR${NC}"
    echo "=================================================="
}

# =============================================================================
# CONFIGURATION DISPLAY
# =============================================================================

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}    EQT DETECTION - PARALLEL DAY PROCESSING${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

# Use default configuration (no interactive input)
echo -e "${GREEN}Using default configuration:${NC}"
display_config
echo ""

# Auto-proceed with default configuration
echo -e "${GREEN}Proceeding with default configuration...${NC}"

# =============================================================================
# VALIDATION
# =============================================================================

echo ""
echo -e "${YELLOW}Validating configuration...${NC}"

# Validate directories
validate_directory "$BASEDIR" "Base" || exit 1
validate_directory "$DATADIR" "Data" || exit 1

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: Python script $PYTHON_SCRIPT not found!${NC}"
    exit 1
fi

# Validate mode
if [ $MODE -lt 1 ] || [ $MODE -gt 3 ]; then
    echo -e "${RED}Error: Mode must be 1, 2, or 3${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Configuration validated successfully${NC}"

# =============================================================================
# GENERATE JULIAN DAYS LIST
# =============================================================================

echo ""
echo -e "${YELLOW}Generating julian days list...${NC}"

# Generate list of dates and julian days
DATE_LIST=()
while IFS=',' read -r year julian_day date_str; do
    if [ -n "$year" ] && [ -n "$julian_day" ] && [ -n "$date_str" ]; then
        DATE_LIST+=("$year,$julian_day,$date_str")
    fi
done < <(generate_julian_days "$START_DATE" "$END_DATE")

if [ ${#DATE_LIST[@]} -eq 0 ]; then
    echo -e "${RED}Error: No valid dates generated. Check your date range.${NC}"
    exit 1
fi

TOTAL_DAYS=${#DATE_LIST[@]}
echo -e "${GREEN}✓ Generated $TOTAL_DAYS days to process${NC}"

# Display the days that will be processed
echo -e "${CYAN}Days to process:${NC}"
for day_info in "${DATE_LIST[@]}"; do
    IFS=',' read -r year julian_day date_str <<< "$day_info"
    echo "  Day $julian_day ($date_str)"
done

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

echo ""
echo -e "${YELLOW}Setting up environment...${NC}"

# Check if conda environment is already active
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" = "$CONDA_ENV" ]; then
    echo -e "${GREEN}✓ Conda environment $CONDA_ENV is active${NC}"
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}⚠️  Current environment: $CONDA_DEFAULT_ENV (expected: $CONDA_ENV)${NC}"
    echo -e "${YELLOW}Continuing anyway...${NC}"
else
    echo -e "${YELLOW}⚠️  No conda environment detected, continuing...${NC}"
fi

# =============================================================================
# PROCESSING SETUP
# =============================================================================

SUCCESSFUL_DAYS=0
FAILED_DAYS=0
FAILED_DAYS_LIST=()

echo ""
echo -e "${YELLOW}Processing setup:${NC}"
echo "Total days to process: $TOTAL_DAYS"
echo "Days per batch: $PARALLEL_JOBS"
echo "Expected batches: $(((TOTAL_DAYS + PARALLEL_JOBS - 1) / PARALLEL_JOBS))"
echo ""

# Start timing
SCRIPT_START_TIME=$(date +%s)
echo -e "${GREEN}Processing started at: $(date)${NC}"
echo ""

# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

# Function to process a single day
process_day() {
    local year="$1"
    local jday="$2"
    local date_str="$3"
    local LOG_FILE="$LOG_DIR/eqt_day_${jday}_${year}.log"
    local DAY_START_TIME=$(date +%s)
    
    echo -e "${YELLOW}[PID $$] Processing day $jday ($date_str)${NC}" | tee -a "$LOG_FILE"
    
    # Run the Python script - let it handle all the date conversions
    python3 "$PYTHON_SCRIPT" \
        --year "$year" \
        --jday "$jday" \
        --basedir "$BASEDIR" \
        --datadir "$DATADIR" \
        --mode "$MODE" >> "$LOG_FILE" 2>&1
    
    local exit_code=$?
    local DAY_END_TIME=$(date +%s)
    local DAY_DURATION=$((DAY_END_TIME - DAY_START_TIME))
    local DAY_MINUTES=$((DAY_DURATION / 60))
    local DAY_SECONDS=$((DAY_DURATION % 60))
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ [PID $$] Day $jday ($date_str) completed successfully in ${DAY_MINUTES}m ${DAY_SECONDS}s${NC}"
        echo "SUCCESS:$jday:$DAY_DURATION:$date_str" >> /tmp/eqt_results_$$.tmp
        
        # Extract pick counts from log file if available
        local max_picks=$(grep -o "Max picks: [0-9]*" "$LOG_FILE" | tail -1 | grep -o "[0-9]*" || echo "0")
        local avg_picks=$(grep -o "Avg picks: [0-9]*" "$LOG_FILE" | tail -1 | grep -o "[0-9]*" || echo "0")
        echo "PICKS:$jday:$max_picks:$avg_picks:$date_str" >> /tmp/eqt_results_$$.tmp
    else
        echo -e "${RED}✗ [PID $$] Day $jday ($date_str) failed after ${DAY_MINUTES}m ${DAY_SECONDS}s (exit code: $exit_code)${NC}"
        echo "FAILED:$jday:$DAY_DURATION:$date_str" >> /tmp/eqt_results_$$.tmp
        echo -e "${RED}Check log file: $LOG_FILE${NC}"
    fi
}

# =============================================================================
# PARALLEL PROCESSING (pool continuo - siempre mantiene N trabajos activos)
# =============================================================================

echo -e "${BLUE}Processing $TOTAL_DAYS days with $PARALLEL_JOBS concurrent jobs...${NC}"
echo ""

# Clean up any previous temp files
rm -f /tmp/eqt_results_*.tmp

# Función para esperar hasta tener espacio para un nuevo trabajo
wait_for_slot() {
    while [ $(jobs -rp | wc -l) -ge $PARALLEL_JOBS ]; do
        sleep 0.5
    done
}

# Procesar cada día, manteniendo siempre PARALLEL_JOBS activos
PROCESSED_COUNT=0
for day_info in "${DATE_LIST[@]}"; do
    IFS=',' read -r year julian_day date_str <<< "$day_info"

    wait_for_slot

    process_day "$year" "$julian_day" "$date_str" &

    PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
    echo -e "${CYAN}Started job $PROCESSED_COUNT/$TOTAL_DAYS: Day $julian_day ($date_str) - Active jobs: $(jobs -rp | wc -l)${NC}"
done

# Esperar a que terminen todos los procesos restantes
echo ""
echo -e "${YELLOW}Waiting for remaining jobs to complete...${NC}"
wait
echo -e "${GREEN}All jobs completed.${NC}"
echo ""

# # =============================================================================
# # PARALLEL PROCESSING
# # =============================================================================

# echo -e "${BLUE}Processing $TOTAL_DAYS days in batches of $PARALLEL_JOBS...${NC}"
# echo ""

# # Clean up any previous temp files
# rm -f /tmp/eqt_results_*.tmp

# # Process days in parallel batches
# for ((batch_start=0; batch_start<${#DATE_LIST[@]}; batch_start+=PARALLEL_JOBS)); do
#     batch_end=$((batch_start + PARALLEL_JOBS - 1))
#     if [ $batch_end -ge ${#DATE_LIST[@]} ]; then
#         batch_end=$((${#DATE_LIST[@]} - 1))
#     fi
    
#     BATCH_START_TIME=$(date +%s)
#     actual_batch_start=$((batch_start + 1))
#     actual_batch_end=$((batch_end + 1))
#     echo -e "${YELLOW}=== Starting batch: days $actual_batch_start to $actual_batch_end ===${NC}"
    
#     # Start parallel processes for this batch
#     for ((i=batch_start; i<=batch_end; i++)); do
#         if [ $i -lt ${#DATE_LIST[@]} ]; then
#             IFS=',' read -r year julian_day date_str <<< "${DATE_LIST[$i]}"
#             process_day "$year" "$julian_day" "$date_str" &
#         fi
#     done
    
#     # Wait for all processes in this batch to complete
#     wait
    
#     BATCH_END_TIME=$(date +%s)
#     BATCH_DURATION=$((BATCH_END_TIME - BATCH_START_TIME))
#     BATCH_MINUTES=$((BATCH_DURATION / 60))
#     BATCH_SECONDS=$((BATCH_DURATION % 60))
    
#     echo -e "${YELLOW}=== Batch completed: days $actual_batch_start to $actual_batch_end in ${BATCH_MINUTES}m ${BATCH_SECONDS}s ===${NC}"
#     echo ""
    
#     # Small delay between batches
#     sleep 1
# done

# =============================================================================
# RESULTS COLLECTION AND SUMMARY
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
TOTAL_MAX_PICKS=0
TOTAL_AVG_PICKS=0

for result_file in /tmp/eqt_results_*.tmp; do
    if [ -f "$result_file" ]; then
        while IFS=':' read -r type jday value1 value2 date_str_result; do
            if [ "$type" = "SUCCESS" ]; then
                ((SUCCESSFUL_DAYS++))
                TOTAL_PROCESSING_TIME=$((TOTAL_PROCESSING_TIME + value1))
            elif [ "$type" = "FAILED" ]; then
                ((FAILED_DAYS++))
                FAILED_DAYS_LIST+=("$jday ($date_str_result)")
                TOTAL_PROCESSING_TIME=$((TOTAL_PROCESSING_TIME + value1))
            elif [ "$type" = "PICKS" ]; then
                TOTAL_MAX_PICKS=$((TOTAL_MAX_PICKS + value1))
                TOTAL_AVG_PICKS=$((TOTAL_AVG_PICKS + value2))
            fi
        done < "$result_file"
    fi
done

# Clean up temporary files
rm -f /tmp/eqt_results_*.tmp

# =============================================================================
# FINAL SUMMARY
# =============================================================================

echo ""
echo "=================================================="
echo -e "${PURPLE}EQT DETECTION PROCESSING COMPLETE${NC}"
echo "=================================================="
echo "Processing finished at: $(date)"
echo -e "Total wall-clock time: ${CYAN}${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s${NC}"
echo -e "Date range processed: ${CYAN}$START_DATE to $END_DATE${NC}"
echo -e "Total days processed: ${CYAN}$TOTAL_DAYS${NC}"
echo -e "Successful days: ${GREEN}$SUCCESSFUL_DAYS${NC}"
echo -e "Failed days: ${RED}$FAILED_DAYS${NC}"

# Display pick statistics
if [ $SUCCESSFUL_DAYS -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}PICK STATISTICS:${NC}"
    if [ $TOTAL_MAX_PICKS -gt 0 ] || [ $TOTAL_AVG_PICKS -gt 0 ]; then
        echo -e "Total max picks detected: ${CYAN}$(printf "%'d" $TOTAL_MAX_PICKS)${NC}"
        echo -e "Total avg picks detected: ${CYAN}$(printf "%'d" $TOTAL_AVG_PICKS)${NC}"
        echo -e "Average max picks per day: ${CYAN}$((TOTAL_MAX_PICKS / SUCCESSFUL_DAYS))${NC}"
        echo -e "Average avg picks per day: ${CYAN}$((TOTAL_AVG_PICKS / SUCCESSFUL_DAYS))${NC}"
    else
        echo -e "${YELLOW}Pick statistics not available (check log files for details)${NC}"
    fi
fi

# Calculate performance statistics
if [ $SUCCESSFUL_DAYS -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}PERFORMANCE STATISTICS:${NC}"
    
    AVG_TIME_PER_DAY=$((TOTAL_PROCESSING_TIME / SUCCESSFUL_DAYS))
    AVG_MINUTES=$((AVG_TIME_PER_DAY / 60))
    AVG_SECONDS=$((AVG_TIME_PER_DAY % 60))
    echo -e "Average processing time per day: ${CYAN}${AVG_MINUTES}m ${AVG_SECONDS}s${NC}"
    
    # Calculate speedup
    ESTIMATED_SEQUENTIAL_TIME=$((AVG_TIME_PER_DAY * SUCCESSFUL_DAYS))
    EST_SEQ_HOURS=$((ESTIMATED_SEQUENTIAL_TIME / 3600))
    EST_SEQ_MINUTES=$(((ESTIMATED_SEQUENTIAL_TIME % 3600) / 60))
    
    if command -v bc &> /dev/null && [ $TOTAL_DURATION -gt 0 ]; then
        SPEEDUP=$(echo "scale=1; $ESTIMATED_SEQUENTIAL_TIME / $TOTAL_DURATION" | bc -l)
        echo -e "Estimated sequential time: ${CYAN}${EST_SEQ_HOURS}h ${EST_SEQ_MINUTES}m${NC}"
        echo -e "Speedup achieved: ${CYAN}${SPEEDUP}x${NC}"
    fi
fi

# Display failed days
if [ $FAILED_DAYS -gt 0 ]; then
    echo ""
    echo -e "${RED}FAILED DAYS:${NC}"
    echo -e "${RED}Failed days: ${FAILED_DAYS_LIST[*]}${NC}"
    echo ""
    echo -e "${YELLOW}You can rerun failed days individually with:${NC}"
    for failed_day in "${FAILED_DAYS_LIST[@]}"; do
        jday=$(echo "$failed_day" | cut -d' ' -f1)
        year=$(echo $START_DATE | cut -d'-' -f1)
        echo "python $PYTHON_SCRIPT --year $year --jday $jday --basedir \"$BASEDIR\" --datadir \"$DATADIR\" --mode $MODE"
    done
else
    echo -e "${GREEN}🎉 All days processed successfully!${NC}"
fi

# Calculate success rate
if [ $TOTAL_DAYS -gt 0 ]; then
    SUCCESS_RATE=$((SUCCESSFUL_DAYS * 100 / TOTAL_DAYS))
    echo ""
    echo -e "Success rate: ${CYAN}$SUCCESS_RATE%${NC}"
fi

echo ""
echo -e "${YELLOW}Output directories:${NC}"
YEAR=$(echo $START_DATE | cut -d'-' -f1)
echo -e "Picks: ${CYAN}/Volumes/GeoPhysics_49/users-data/montalca/CATALOGS/EQT_PICKS/PICKS/$YEAR/${NC}"
echo -e "Probabilities: ${CYAN}/Volumes/GeoPhysics_49/users-data/montalca/CATALOGS/EQT_PICKS/PROBS/$YEAR/${NC}"
echo -e "Logs: ${CYAN}$LOG_DIR/eqt_day_*_$YEAR.log${NC}"

echo "=================================================="

# Exit with appropriate code
if [ $FAILED_DAYS -eq 0 ]; then
    echo -e "${GREEN}🎉 EQT Detection completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  EQT Detection completed with some failures.${NC}"
    exit 1
fi