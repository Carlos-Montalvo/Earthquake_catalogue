#!/bin/bash

# PYOCTO ASSOCIATION Script - Simple parallel execution of PyOcto_association_single_day.py
# This script only handles parallelization, all date logic is in the Python script

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Time period parameters (YYYY-MM-DD format)
START_DATE="2025-07-01" #2025-05-09
END_DATE="2025-09-30"  # Adjust date range as needed

# Directory parameters
BASEDIR="/Volumes/GeoPhysics_49/users-data/montalca"
DATADIR="DATA"

# Processing parameters
PARALLEL_JOBS=6  # Number of days to process in parallel
PICKS_TYPE="kurtosis"  # Options: max, avg, kurtosis

# Python environment
PYTHON_SCRIPT="/Volumes/GeoPhysics_49/users-data/montalca/PROGRAMS/PYTHON/EARTHQUAKE_CATALOGUE/PyOcto_association_single_day.py"
PYTHON_EXECUTABLE="/home/montalca/miniconda3/envs/seisbench/bin/python"
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
    echo -e "${PURPLE}PYOCTO ASSOCIATION CONFIGURATION${NC}"
    echo "=================================================="
    echo -e "Start date: ${YELLOW}$START_DATE${NC}"
    echo -e "End date: ${YELLOW}$END_DATE${NC}"
    echo -e "Base directory: ${YELLOW}$BASEDIR${NC}"
    echo -e "Data directory: ${YELLOW}$DATADIR${NC}"
    echo -e "Picks type: ${YELLOW}$PICKS_TYPE${NC}"
    echo -e "Parallel jobs: ${YELLOW}$PARALLEL_JOBS${NC}"
    echo -e "Python script: ${YELLOW}$PYTHON_SCRIPT${NC}"
    echo -e "Python executable: ${YELLOW}$PYTHON_EXECUTABLE${NC}"
    echo -e "Conda environment: ${YELLOW}$CONDA_ENV${NC}"
    echo -e "Log directory: ${YELLOW}$LOG_DIR${NC}"
    echo "=================================================="
}

# =============================================================================
# CONFIGURATION DISPLAY
# =============================================================================

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}    PYOCTO ASSOCIATION - PARALLEL DAY PROCESSING${NC}"
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
validate_directory "$BASEDIR/$DATADIR" "Data" || exit 1

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: Python script $PYTHON_SCRIPT not found!${NC}"
    exit 1
fi

# Validate picks type
if [[ ! "$PICKS_TYPE" =~ ^(max|avg|kurtosis)$ ]]; then
    echo -e "${RED}Error: PICKS_TYPE must be one of: max, avg, kurtosis${NC}"
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
echo "Concurrent jobs: $PARALLEL_JOBS"
echo "Picks type: $PICKS_TYPE"
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
    local LOG_FILE="$LOG_DIR/pyocto_day_${jday}_${year}.log"
    local DAY_START_TIME=$(date +%s)
    
    echo -e "${YELLOW}[PID $$] Processing day $jday ($date_str)${NC}" | tee -a "$LOG_FILE"
    
    # Use direct Python executable path to avoid conda activation issues
    "$PYTHON_EXECUTABLE" "$PYTHON_SCRIPT" \
        "$BASEDIR" \
        "$DATADIR" \
        "$year" \
        "$jday" \
        "$PICKS_TYPE" >> "$LOG_FILE" 2>&1
    
    local exit_code=$?
    local DAY_END_TIME=$(date +%s)
    local DAY_DURATION=$((DAY_END_TIME - DAY_START_TIME))
    local DAY_MINUTES=$((DAY_DURATION / 60))
    local DAY_SECONDS=$((DAY_DURATION % 60))
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ [PID $$] Day $jday ($date_str) completed successfully in ${DAY_MINUTES}m ${DAY_SECONDS}s${NC}"
        echo "SUCCESS:$jday:$DAY_DURATION:$date_str" >> /tmp/pyocto_results_$$.tmp
        
        # Extract event counts from log file if available
        local events_count=$(grep -o "Events found: [0-9]*" "$LOG_FILE" | tail -1 | grep -o "[0-9]*" || echo "0")
        local assignments_count=$(grep -o "Assignments made: [0-9]*" "$LOG_FILE" | tail -1 | grep -o "[0-9]*" || echo "0")
        echo "EVENTS:$jday:$events_count:$assignments_count:$date_str" >> /tmp/pyocto_results_$$.tmp
    else
        echo -e "${RED}✗ [PID $$] Day $jday ($date_str) failed after ${DAY_MINUTES}m ${DAY_SECONDS}s (exit code: $exit_code)${NC}"
        echo "FAILED:$jday:$DAY_DURATION:$date_str" >> /tmp/pyocto_results_$$.tmp
        echo -e "${RED}Check log file: $LOG_FILE${NC}"
    fi
}

# =============================================================================
# PARALLEL PROCESSING (pool continuo - siempre mantiene N trabajos activos)
# =============================================================================

echo -e "${BLUE}Processing $TOTAL_DAYS days with $PARALLEL_JOBS concurrent jobs...${NC}"
echo ""

# Clean up any previous temp files
rm -f /tmp/pyocto_results_*.tmp

# Función para esperar hasta tener espacio para un nuevo trabajo
# Mantiene siempre PARALLEL_JOBS trabajos activos
wait_for_slot() {
    while [ $(jobs -rp | wc -l) -ge $PARALLEL_JOBS ]; do
        sleep 0.5
    done
}

# Procesar cada día, manteniendo siempre PARALLEL_JOBS activos
PROCESSED_COUNT=0
for day_info in "${DATE_LIST[@]}"; do
    IFS=',' read -r year julian_day date_str <<< "$day_info"
    
    # Esperar hasta que haya un slot disponible
    wait_for_slot
    
    # Iniciar el proceso en background
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

# =============================================================================
# ORIGINAL BATCH PROCESSING (comentado)
# =============================================================================
# El código original procesaba por batches, esperando a que termine cada batch
# completo antes de iniciar el siguiente. La nueva implementación mantiene
# siempre N trabajos activos (pool continuo).
#
# # Process days in parallel batches
# for ((batch_start=0; batch_start<${#DATE_LIST[@]}; batch_start+=PARALLEL_JOBS)); do
#     batch_end=$((batch_start + PARALLEL_JOBS - 1))
#     if [ $batch_end -ge ${#DATE_LIST[@]} ]; then
#         batch_end=$((${#DATE_LIST[@]} - 1))
#     fi
#     
#     BATCH_START_TIME=$(date +%s)
#     actual_batch_start=$((batch_start + 1))
#     actual_batch_end=$((batch_end + 1))
#     echo -e "${YELLOW}=== Starting batch: days $actual_batch_start to $actual_batch_end ===${NC}"
#     
#     # Start parallel processes for this batch
#     for ((i=batch_start; i<=batch_end; i++)); do
#         if [ $i -lt ${#DATE_LIST[@]} ]; then
#             IFS=',' read -r year julian_day date_str <<< "${DATE_LIST[$i]}"
#             process_day "$year" "$julian_day" "$date_str" &
#         fi
#     done
#     
#     # Wait for all processes in this batch to complete
#     wait
#     
#     BATCH_END_TIME=$(date +%s)
#     BATCH_DURATION=$((BATCH_END_TIME - BATCH_START_TIME))
#     BATCH_MINUTES=$((BATCH_DURATION / 60))
#     BATCH_SECONDS=$((BATCH_DURATION % 60))
#     
#     echo -e "${YELLOW}=== Batch completed: days $actual_batch_start to $actual_batch_end in ${BATCH_MINUTES}m ${BATCH_SECONDS}s ===${NC}"
#     echo ""
# done

# =============================================================================
# RESULTS COLLECTION AND SUMMARY
# =============================================================================

echo -e "${BLUE}Collecting results...${NC}"

# Collect results from all temporary files
for tmp_file in /tmp/pyocto_results_*.tmp; do
    if [ -f "$tmp_file" ]; then
        while IFS=':' read -r status jday duration events_or_extra date_str extra; do
            case $status in
                "SUCCESS")
                    SUCCESSFUL_DAYS=$((SUCCESSFUL_DAYS + 1))
                    ;;
                "FAILED")
                    FAILED_DAYS=$((FAILED_DAYS + 1))
                    FAILED_DAYS_LIST+=("$jday ($date_str)")
                    ;;
                "EVENTS")
                    # Store events info for summary
                    ;;
            esac
        done < "$tmp_file"
    fi
done

# Clean up temp files
rm -f /tmp/pyocto_results_*.tmp

# =============================================================================
# FINAL SUMMARY
# =============================================================================

SCRIPT_END_TIME=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}    PYOCTO ASSOCIATION COMPLETED${NC}"
echo -e "${BLUE}==================================================${NC}"
echo -e "Total days processed: ${YELLOW}$TOTAL_DAYS${NC}"
echo -e "Successful days: ${GREEN}$SUCCESSFUL_DAYS${NC}"
echo -e "Failed days: ${RED}$FAILED_DAYS${NC}"
echo -e "Picks type used: ${YELLOW}$PICKS_TYPE${NC}"
echo -e "Total processing time: ${YELLOW}${TOTAL_MINUTES}m ${TOTAL_SECONDS}s${NC}"
echo -e "Processing finished at: ${YELLOW}$(date)${NC}"

if [ $FAILED_DAYS -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed days:${NC}"
    for failed_day in "${FAILED_DAYS_LIST[@]}"; do
        echo -e "  ${RED}✗ $failed_day${NC}"
    done
fi

echo ""
echo -e "${CYAN}Results are saved in:${NC}"
echo -e "  ${YELLOW}Events: /Volumes/GeoPhysics_49/users-data/montalca/CATALOGS/PYOCTO/EVENTS/${NC}"
echo -e "  ${YELLOW}Assignments: /Volumes/GeoPhysics_49/users-data/montalca/CATALOGS/PYOCTO/ASSIGNMENTS/${NC}"
echo -e "  ${YELLOW}NonLinLoc: /Volumes/GeoPhysics_49/users-data/montalca/NLL/${NC}"
echo ""
echo -e "${CYAN}Log files are in:${NC}"
echo -e "  ${YELLOW}$LOG_DIR${NC}"
echo ""

# =============================================================================
# EXIT
# =============================================================================

if [ $FAILED_DAYS -eq 0 ]; then
    echo -e "${GREEN}✓ All days processed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some days failed. Check log files for details.${NC}"
    exit 1
fi
