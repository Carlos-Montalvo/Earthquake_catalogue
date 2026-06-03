#!/bin/bash

# NONLINLOC LOCATION Script - Daily processing of seismic events
# This script modifies NLL config files for each day and runs NonLinLoc location

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Time period parameters (YYYY-MM-DD format)
START_DATE="2025-03-01"
END_DATE="2025-06-30"  # Adjust date range as needed

# Directory parameters
BASEDIR="/Volumes/GeoPhysics_49/users-data/montalca"
DATADIR="DATA"
NLL_DIR="$BASEDIR/NLL"
GRIDS_DIR="$NLL_DIR/IN/GRIDS"  # Directory containing TIME and VEL subdirs
TIME_GRIDS_DIR="$GRIDS_DIR/TIME"  # Time grids directory
VEL_GRIDS_DIR="$GRIDS_DIR/VEL"    # Velocity grids directory
PICKS_DIR="$NLL_DIR/PYOCTO_ASSIGNMENTS"  # Directory with .obs files
CONFIG_TEMPLATE="$NLL_DIR/CONFIG_FILE_TEMPLATE.in"  # Template config file
RESULTS_DIR="$NLL_DIR/OUT_JAN25_SEP25"  # Output directory for location results

# Processing parameters
PARALLEL_JOBS=10  # Number of days to process in parallel
NLL_EXECUTABLE="NLLoc"  # NonLinLoc executable name

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

# Function to validate file
validate_file() {
    local file="$1"
    local name="$2"
    
    if [ ! -f "$file" ]; then
        echo -e "${RED}Error: $name file '$file' does not exist!${NC}"
        return 1
    fi
    return 0
}

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
    echo -e "${PURPLE}NONLINLOC LOCATION CONFIGURATION${NC}"
    echo "=================================================="
    echo -e "Start date: ${YELLOW}$START_DATE${NC}"
    echo -e "End date: ${YELLOW}$END_DATE${NC}"
    echo -e "Base directory: ${YELLOW}$BASEDIR${NC}"
    echo -e "NLL directory: ${YELLOW}$NLL_DIR${NC}"
    echo -e "Grids directory: ${YELLOW}$GRIDS_DIR${NC}"
    echo -e "Time grids: ${YELLOW}$TIME_GRIDS_DIR${NC}"
    echo -e "Velocity grids: ${YELLOW}$VEL_GRIDS_DIR${NC}"
    echo -e "Picks directory: ${YELLOW}$PICKS_DIR${NC}"
    echo -e "Config template: ${YELLOW}$CONFIG_TEMPLATE${NC}"
    echo -e "Results directory: ${YELLOW}$RESULTS_DIR${NC}"
    echo -e "Parallel jobs: ${YELLOW}$PARALLEL_JOBS${NC}"
    echo -e "NLL executable: ${YELLOW}$NLL_EXECUTABLE${NC}"
    echo -e "Log directory: ${YELLOW}$LOG_DIR${NC}"
    echo "=================================================="
}

# Function to create config file for specific day
create_daily_config() {
    local year="$1"
    local jday="$2"
    local date_str="$3"
    
    local obs_file="$PICKS_DIR/${DATADIR}_${year}_${jday}.obs"
    local config_file="$NLL_DIR/nlloc_${year}_${jday}.in"
    local output_prefix="${DATADIR}_${year}_${jday}"
    
    # Check if observation file exists
    if [ ! -f "$obs_file" ]; then
        echo -e "${YELLOW}Warning: Observation file not found: $obs_file${NC}"
        return 1
    fi
    
    # Create config file from template
    if [ -f "$CONFIG_TEMPLATE" ]; then
        # Replace the LOCFILES line in the template to use the specific obs file and output
        # Original line: LOCFILES IN/PICKS/*.nll NLLOC_OBS IN/GRIDS/TIME/TIME_GRID OUT/located
        # Replace with specific file paths for this day
        sed -e "s|LOCFILES IN/PICKS/\*\.nll NLLOC_OBS IN/GRIDS/TIME/TIME_GRID OUT/located|LOCFILES $obs_file NLLOC_OBS IN/GRIDS/TIME/TIME_GRID $RESULTS_DIR/$output_prefix|g" \
            "$CONFIG_TEMPLATE" > "$config_file"
    else
        echo -e "${RED}Error: Template file not found: $CONFIG_TEMPLATE${NC}"
        return 1
    fi
    
    echo "$config_file"
}

# =============================================================================
# CONFIGURATION DISPLAY
# =============================================================================

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}    NONLINLOC LOCATION - DAILY PROCESSING${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

display_config
echo ""

# =============================================================================
# VALIDATION
# =============================================================================

echo ""
echo -e "${YELLOW}Validating configuration...${NC}"

# Validate directories
validate_directory "$BASEDIR" "Base" || exit 1
validate_directory "$NLL_DIR" "NLL" || exit 1
validate_directory "$GRIDS_DIR" "Grids" || exit 1
validate_directory "$TIME_GRIDS_DIR" "Time grids" || exit 1
validate_directory "$VEL_GRIDS_DIR" "Velocity grids" || exit 1
validate_directory "$PICKS_DIR" "Picks" || exit 1

# Validate template file
validate_file "$CONFIG_TEMPLATE" "Config template" || exit 1

# Create results directory if it doesn't exist
if [ ! -d "$RESULTS_DIR" ]; then
    mkdir -p "$RESULTS_DIR"
    echo -e "${GREEN}Created results directory: $RESULTS_DIR${NC}"
fi

# Activate NonLinLoc environment (commented out - activate manually with 'need nll')
# echo -e "${YELLOW}Activating NonLinLoc environment...${NC}"
# need nll
# if [ $? -ne 0 ]; then
#     echo -e "${RED}Error: Failed to activate NonLinLoc with 'need nll'${NC}"
#     exit 1
# fi

# Check if NonLinLoc executable is available
if ! command -v "$NLL_EXECUTABLE" &> /dev/null; then
    echo -e "${RED}Error: NonLinLoc executable '$NLL_EXECUTABLE' not found in PATH${NC}"
    echo -e "${YELLOW}Make sure NonLinLoc is installed and in your PATH${NC}"
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
    local LOG_FILE="$LOG_DIR/nlloc_day_${jday}_${year}.log"
    local DAY_START_TIME=$(date +%s)
    
    echo -e "${YELLOW}[PID $$] Processing day $jday ($date_str)${NC}" | tee -a "$LOG_FILE"
    
    # Create config file for this day
    local config_file=$(create_daily_config "$year" "$jday" "$date_str")
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ [PID $$] Failed to create config for day $jday ($date_str)${NC}"
        echo "FAILED:$jday:0:$date_str" >> /tmp/nlloc_results_$$.tmp
        return 1
    fi
    
    # Run NonLinLoc (change to NLL directory first for relative paths)
    echo "Running NonLinLoc for day $jday..." >> "$LOG_FILE"
    cd "$NLL_DIR" && "$NLL_EXECUTABLE" "$config_file" >> "$LOG_FILE" 2>&1
    
    local exit_code=$?
    local DAY_END_TIME=$(date +%s)
    local DAY_DURATION=$((DAY_END_TIME - DAY_START_TIME))
    local DAY_MINUTES=$((DAY_DURATION / 60))
    local DAY_SECONDS=$((DAY_DURATION % 60))
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ [PID $$] Day $jday ($date_str) completed successfully in ${DAY_MINUTES}m ${DAY_SECONDS}s${NC}"
        echo "SUCCESS:$jday:$DAY_DURATION:$date_str" >> /tmp/nlloc_results_$$.tmp
        
        # Count located events if possible
        local hyp_file="$RESULTS_DIR/${DATADIR}_${year}_${jday}.hyp"
        if [ -f "$hyp_file" ]; then
            local events_count=$(grep -c "^GEOGRAPHIC" "$hyp_file" 2>/dev/null || echo "0")
            echo "EVENTS:$jday:$events_count:$date_str" >> /tmp/nlloc_results_$$.tmp
        fi
        
        # Clean up config file
        rm -f "$config_file"
    else
        echo -e "${RED}✗ [PID $$] Day $jday ($date_str) failed after ${DAY_MINUTES}m ${DAY_SECONDS}s (exit code: $exit_code)${NC}"
        echo "FAILED:$jday:$DAY_DURATION:$date_str" >> /tmp/nlloc_results_$$.tmp
        echo -e "${RED}Check log file: $LOG_FILE${NC}"
        # Keep config file for debugging
        echo -e "${YELLOW}Config file kept for debugging: $config_file${NC}"
    fi
}

# # =============================================================================
# # PARALLEL PROCESSING (pool continuo - siempre mantiene N trabajos activos)
# # =============================================================================

echo -e "${BLUE}Processing $TOTAL_DAYS days with $PARALLEL_JOBS concurrent jobs...${NC}"
echo ""

# Clean up any previous temp files
rm -f /tmp/nlloc_results_*.tmp

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

# # =============================================================================
# # ORIGINAL BATCH PROCESSING (comentado)
# # =============================================================================
# # El código original procesaba por batches, esperando a que termine cada batch
# # completo antes de iniciar el siguiente. La nueva implementación mantiene
# # siempre N trabajos activos (pool continuo).

# # Process days in parallel batches
# for ((batch_start=0; batch_start<${#DATE_LIST[@]}; batch_start+=PARALLEL_JOBS)); do
#    batch_end=$((batch_start + PARALLEL_JOBS - 1))
#    if [ $batch_end -ge ${#DATE_LIST[@]} ]; then
#        batch_end=$((${#DATE_LIST[@]} - 1))
#    fi
   
#    BATCH_START_TIME=$(date +%s)
#    actual_batch_start=$((batch_start + 1))
#    actual_batch_end=$((batch_end + 1))
#    echo -e "${YELLOW}=== Starting batch: days $actual_batch_start to $actual_batch_end ===${NC}"
   
#    # Start parallel processes for this batch
#    for ((i=batch_start; i<=batch_end; i++)); do
#        if [ $i -lt ${#DATE_LIST[@]} ]; then
#            IFS=',' read -r year julian_day date_str <<< "${DATE_LIST[$i]}"
#            process_day "$year" "$julian_day" "$date_str" &
#        fi
#    done
   
#    # Wait for all processes in this batch to complete
#    wait
   
#    BATCH_END_TIME=$(date +%s)
#    BATCH_DURATION=$((BATCH_END_TIME - BATCH_START_TIME))
#    BATCH_MINUTES=$((BATCH_DURATION / 60))
#    BATCH_SECONDS=$((BATCH_DURATION % 60))
   
#    echo -e "${YELLOW}=== Batch completed: days $actual_batch_start to $actual_batch_end in ${BATCH_MINUTES}m ${BATCH_SECONDS}s ===${NC}"
#    echo ""
# done

# =============================================================================
# RESULTS COLLECTION AND SUMMARY
# =============================================================================

echo -e "${BLUE}Collecting results...${NC}"

TOTAL_LOCATED_EVENTS=0

# Collect results from all temporary files
for tmp_file in /tmp/nlloc_results_*.tmp; do
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
                    TOTAL_LOCATED_EVENTS=$((TOTAL_LOCATED_EVENTS + events_or_extra))
                    ;;
            esac
        done < "$tmp_file"
    fi
done

# Clean up temp files
rm -f /tmp/nlloc_results_*.tmp

# =============================================================================
# FINAL SUMMARY
# =============================================================================

SCRIPT_END_TIME=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}    NONLINLOC LOCATION COMPLETED${NC}"
echo -e "${BLUE}==================================================${NC}"
echo -e "Total days processed: ${YELLOW}$TOTAL_DAYS${NC}"
echo -e "Successful days: ${GREEN}$SUCCESSFUL_DAYS${NC}"
echo -e "Failed days: ${RED}$FAILED_DAYS${NC}"
echo -e "Total located events: ${YELLOW}$TOTAL_LOCATED_EVENTS${NC}"
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
echo -e "  ${YELLOW}Location files: $RESULTS_DIR${NC}"
echo -e "  ${YELLOW}Log files: $LOG_DIR${NC}"
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
