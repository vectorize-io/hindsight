#!/bin/bash
# Start split Ollama execution lanes for Hindsight
# - Port 11434: Embeddings lane (primary, should already be running)
# - Port 11435: LLM/reasoning lane (secondary, started by this script)

set -e

# Configuration
EMBEDDINGS_PORT="${OLLAMA_EMBEDDINGS_PORT:-11434}"
LLM_PORT="${OLLAMA_LLM_PORT:-11435}"
OLLAMA_BINARY="${OLLAMA_BINARY:-/opt/homebrew/opt/ollama/bin/ollama}"
MODELS_DIR="${OLLAMA_MODELS_DIR:-/Volumes/Mac/Users/oliververmeulen/.ollama/models}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Hindsight Split Ollama Lanes - Startup"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if Ollama binary exists
if [ ! -f "$OLLAMA_BINARY" ]; then
    echo -e "${RED}✗ Ollama binary not found at: $OLLAMA_BINARY${NC}"
    echo "  Set OLLAMA_BINARY environment variable to override"
    exit 1
fi

# Check if models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo -e "${RED}✗ Models directory not found at: $MODELS_DIR${NC}"
    echo "  Set OLLAMA_MODELS_DIR environment variable to override"
    exit 1
fi

echo -e "${BLUE}Configuration:${NC}"
echo "  • Ollama Binary: $OLLAMA_BINARY"
echo "  • Models Directory: $MODELS_DIR"
echo "  • Embeddings Port: $EMBEDDINGS_PORT"
echo "  • LLM Port: $LLM_PORT"
echo ""

# Function to check if Ollama is running on a port
check_ollama_port() {
    local port=$1
    curl -s "http://localhost:$port/api/tags" > /dev/null 2>&1
    return $?
}

# Function to get model count on a port
get_model_count() {
    local port=$1
    local count=$(curl -s "http://localhost:$port/api/tags" 2>/dev/null | grep -o '"name"' | wc -l | tr -d ' ')
    echo "$count"
}

# Check primary embeddings lane (11434)
echo -e "${BLUE}Checking Embeddings Lane (port $EMBEDDINGS_PORT)...${NC}"
if check_ollama_port "$EMBEDDINGS_PORT"; then
    model_count=$(get_model_count "$EMBEDDINGS_PORT")
    echo -e "${GREEN}✓ Embeddings lane is running${NC}"
    echo "  • Models available: $model_count"
    
    # Verify nomic-embed-text exists
    if curl -s "http://localhost:$EMBEDDINGS_PORT/api/tags" 2>/dev/null | grep -q "nomic-embed-text"; then
        echo -e "${GREEN}  ✓ nomic-embed-text model found${NC}"
    else
        echo -e "${YELLOW}  ⚠ nomic-embed-text model not found${NC}"
        echo "    Run: ollama pull nomic-embed-text"
    fi
else
    echo -e "${RED}✗ Embeddings lane is NOT running on port $EMBEDDINGS_PORT${NC}"
    echo "  Primary Ollama instance should be running system-wide"
    echo "  Start it with: ollama serve"
    exit 1
fi

echo ""

# Check secondary LLM lane (11435)
echo -e "${BLUE}Checking LLM Lane (port $LLM_PORT)...${NC}"
if check_ollama_port "$LLM_PORT"; then
    model_count=$(get_model_count "$LLM_PORT")
    echo -e "${GREEN}✓ LLM lane is already running${NC}"
    echo "  • Models available: $model_count"
    
    # Verify llama3.2 exists
    if curl -s "http://localhost:$LLM_PORT/api/tags" 2>/dev/null | grep -q "llama3.2"; then
        echo -e "${GREEN}  ✓ llama3.2 model found${NC}"
    else
        echo -e "${YELLOW}  ⚠ llama3.2 model not found${NC}"
        echo "    Run: OLLAMA_HOST=127.0.0.1:$LLM_PORT ollama pull llama3.2"
    fi
else
    echo -e "${YELLOW}⚠ LLM lane is NOT running, starting it now...${NC}"
    
    # Start secondary Ollama instance
    OLLAMA_HOST="127.0.0.1:$LLM_PORT" \
    OLLAMA_MODELS="$MODELS_DIR" \
    nohup "$OLLAMA_BINARY" serve > /tmp/ollama-llm-lane.log 2>&1 &
    
    LLM_PID=$!
    echo "  • Started Ollama on port $LLM_PORT (PID: $LLM_PID)"
    echo "  • Log file: /tmp/ollama-llm-lane.log"
    
    # Wait for it to be ready
    echo -n "  • Waiting for LLM lane to be ready"
    for i in {1..30}; do
        if check_ollama_port "$LLM_PORT"; then
            echo ""
            echo -e "${GREEN}  ✓ LLM lane is ready${NC}"
            
            model_count=$(get_model_count "$LLM_PORT")
            echo "  • Models available: $model_count"
            
            # Verify llama3.2 exists
            if curl -s "http://localhost:$LLM_PORT/api/tags" 2>/dev/null | grep -q "llama3.2"; then
                echo -e "${GREEN}  ✓ llama3.2 model found${NC}"
            else
                echo -e "${YELLOW}  ⚠ llama3.2 model not found${NC}"
                echo "    Run: OLLAMA_HOST=127.0.0.1:$LLM_PORT ollama pull llama3.2"
            fi
            break
        fi
        echo -n "."
        sleep 1
        
        if [ $i -eq 30 ]; then
            echo ""
            echo -e "${RED}✗ LLM lane failed to start after 30 seconds${NC}"
            echo "  Check log: /tmp/ollama-llm-lane.log"
            exit 1
        fi
    done
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo -e "${GREEN}Split Ollama Lanes - READY${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Endpoints:"
echo "  • Embeddings: http://localhost:$EMBEDDINGS_PORT"
echo "  • LLM:        http://localhost:$LLM_PORT"
echo ""
echo "Configuration in .env:"
echo "  HINDSIGHT_API_EMBEDDINGS_OLLAMA_API_BASE=http://localhost:$EMBEDDINGS_PORT"
echo "  HINDSIGHT_API_LLM_OLLAMA_API_BASE=http://localhost:$LLM_PORT"
echo ""
