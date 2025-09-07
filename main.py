from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import PyPDF2
import re
import io
import json
import pandas as pd
import os
from typing import Dict, List, TypedDict, Optional
import logging
from langgraph.graph import StateGraph, START, END
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Court Document Processor with LangGraph")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the state for LangGraph
class DocumentState(TypedDict):
    filename: str
    raw_text: str
    national_id: Optional[str]
    action: Optional[str]
    confidence: float
    processing_steps: List[str]
    errors: List[str]
    customer_valid: bool
    customer_info: Optional[Dict]
    workflow_completed: bool
    rejected: bool
    rejection_reason: Optional[str]

class CustomerDatabase:
    """Handles customer database operations"""
    
    def __init__(self, csv_file_path: str = "customers.csv"):
        self.csv_file_path = csv_file_path
        self.customers_df = None
        self.load_database()
    
    def load_database(self):
        """Load customer database from CSV file"""
        try:
            if os.path.exists(self.csv_file_path):
                self.customers_df = pd.read_csv(self.csv_file_path)
                # Clean column names (remove whitespace)
                self.customers_df.columns = self.customers_df.columns.str.strip()
                logger.info(f"Loaded {len(self.customers_df)} customers from {self.csv_file_path}")
            else:
                logger.error(f"Customer database file {self.csv_file_path} not found!")
                raise FileNotFoundError(f"Customer database file {self.csv_file_path} does not exist")
        except Exception as e:
            logger.error(f"Error loading customer database: {str(e)}")
            raise e
    
    def validate_customer(self, national_id: str) -> Dict:
        """Validate if customer exists in database"""
        if self.customers_df is None:
            return {"valid": False, "reason": "Customer database not available"}
        
        # Convert national_id to string for comparison (in case CSV has it as integer)
        self.customers_df['national_id'] = self.customers_df['national_id'].astype(str)
        
        customer = self.customers_df[self.customers_df['national_id'] == national_id]
        
        if customer.empty:
            return {
                "valid": False,
                "reason": f"Customer with National ID {national_id} not found in database"
            }
        
        customer_data = customer.iloc[0].to_dict()
        
        return {
            "valid": True,
            "customer_info": customer_data,
            "reason": "Customer found and validated"
        }
    
    def get_database_info(self) -> Dict:
        """Get database information"""
        if self.customers_df is None:
            return {"total_customers": 0, "customers": []}
        
        # Convert to records with only the actual fields
        customers_list = []
        for _, row in self.customers_df.iterrows():
            customer_record = row.to_dict()
            customers_list.append(customer_record)
        
        return {
            "total_customers": len(self.customers_df),
            "customers": customers_list
        }

class ActionProcessor:
    """Handles action execution for valid customers"""
    
    SUPPORTED_ACTIONS = {
        "freeze_funds": "Freeze all funds in the customer account",
        "release_funds": "Release all held or restricted funds back to the customer account",
        "suspend_account": "Suspend customer account temporarily",
        "transfer_funds": "Transfer funds as per court order"
    }
    
    @staticmethod
    def execute_action(action: str, customer_info: Dict) -> Dict:
        """Execute the determined action (dummy implementation)"""
        try:
            national_id = customer_info.get('national_id')
            customer_id = customer_info.get('customer_id')
            
            if action == "freeze_funds":
                return {
                    "success": True,
                    "message": f"✅ Funds frozen for customer ID: {customer_id} (National ID: {national_id})",
                    "action_details": "All account funds have been frozen per court order"
                }
            elif action == "release_funds":
                return {
                    "success": True,
                    "message": f"✅ Funds released for customer ID: {customer_id} (National ID: {national_id})",
                    "action_details": "All frozen funds have been released back to customer account"
                }
            elif action == "suspend_account":
                return {
                    "success": True,
                    "message": f"✅ Account suspended for customer ID: {customer_id} (National ID: {national_id})",
                    "action_details": "Customer account has been temporarily suspended"
                }
            elif action == "transfer_funds":
                return {
                    "success": True,
                    "message": f"✅ Fund transfer initiated for customer ID: {customer_id} (National ID: {national_id})",
                    "action_details": "Fund transfer process has been initiated per court order"
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Unknown action: {action}",
                    "action_details": "Action not recognized or supported"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Action execution failed: {str(e)}",
                "action_details": "Error occurred during action execution"
            }

class DocumentProcessingAgent:
    """LangGraph-based document processing agent with customer validation"""
    
    def __init__(self, customer_db: CustomerDatabase):
        self.customer_db = customer_db
        self.action_processor = ActionProcessor()
        self.graph = self._build_processing_graph()
        self.id_pattern = r'\b\d{10}\b'
        
    def _build_processing_graph(self) -> StateGraph:
        """Build the LangGraph processing workflow with customer validation"""
        
        # Create the graph
        workflow = StateGraph(DocumentState)
        
        # Add nodes (processing steps)
        workflow.add_node("extract_text", self._extract_text_node)
        workflow.add_node("identify_national_id", self._identify_national_id_node)
        workflow.add_node("validate_customer", self._validate_customer_node)
        workflow.add_node("determine_action", self._determine_action_node)
        workflow.add_node("execute_action", self._execute_action_node)
        workflow.add_node("finalize_success", self._finalize_success_node)
        workflow.add_node("finalize_rejection", self._finalize_rejection_node)
        
        # Define the workflow edges with conditional routing
        workflow.add_edge("extract_text", "identify_national_id")
        workflow.add_edge("identify_national_id", "validate_customer")
        
        # Conditional routing based on customer validation
        workflow.add_conditional_edges(
            "validate_customer",
            self._should_process_customer,
            {
                "process": "determine_action",
                "reject": "finalize_rejection"
            }
        )
        
        workflow.add_edge("determine_action", "execute_action")
        workflow.add_edge("execute_action", "finalize_success")
        workflow.add_edge("finalize_success", END)
        workflow.add_edge("finalize_rejection", END)
        
        # Set entry point
        workflow.set_entry_point("extract_text")
        
        return workflow.compile()
    
    def _should_process_customer(self, state: DocumentState) -> str:
        """Conditional function to determine if customer should be processed"""
        if state.get("customer_valid") and state.get("national_id"):
            return "process"
        else:
            return "reject"
    
    def _extract_text_node(self, state: DocumentState) -> DocumentState:
        """Node 1: Extract text from document"""
        try:
            state["processing_steps"].append("📄 Step 1: Extracting text from document")
            
            if state["filename"].lower().endswith('.pdf'):
                state["processing_steps"].append("✅ PDF text extraction completed")
            else:
                state["processing_steps"].append("✅ Plain text document processed")
                
            state["confidence"] = 0.9 if state["raw_text"] else 0.1
            
        except Exception as e:
            state["errors"].append(f"Text extraction failed: {str(e)}")
            state["confidence"] = 0.0
            
        return state
    
    def _identify_national_id_node(self, state: DocumentState) -> DocumentState:
        """Node 2: Identify National ID using pattern matching"""
        try:
            state["processing_steps"].append("🔍 Step 2: Searching for National ID patterns")
            
            # Extract National ID using regex
            matches = re.findall(self.id_pattern, state["raw_text"])
            
            if matches:
                state["national_id"] = matches[0]
                state["processing_steps"].append(f"✅ Found National ID: {state['national_id']}")
                state["confidence"] *= 0.95
            else:
                state["national_id"] = None
                state["processing_steps"].append("❌ No National ID found")
                state["errors"].append("National ID not detected in document")
                state["confidence"] *= 0.5
                
        except Exception as e:
            state["errors"].append(f"ID identification failed: {str(e)}")
            state["confidence"] *= 0.3
            
        return state
    
    def _validate_customer_node(self, state: DocumentState) -> DocumentState:
        """Node 3: Validate customer against database"""
        try:
            state["processing_steps"].append("🏛️ Step 3: Validating customer in database")
            
            if not state.get("national_id"):
                state["customer_valid"] = False
                state["rejection_reason"] = "No National ID found to validate"
                state["processing_steps"].append("❌ Cannot validate customer - No National ID")
                return state
            
            validation_result = self.customer_db.validate_customer(state["national_id"])
            
            if validation_result["valid"]:
                state["customer_valid"] = True
                state["customer_info"] = validation_result["customer_info"]
                customer_id = state["customer_info"].get("customer_id", "Unknown")
                state["processing_steps"].append(f"✅ Customer validated: Customer ID {customer_id}")
                state["confidence"] *= 0.95
            else:
                state["customer_valid"] = False
                state["rejection_reason"] = validation_result["reason"]
                state["processing_steps"].append(f"❌ Customer validation failed: {validation_result['reason']}")
                state["confidence"] *= 0.2
                
        except Exception as e:
            state["errors"].append(f"Customer validation failed: {str(e)}")
            state["customer_valid"] = False
            state["confidence"] *= 0.1
            
        return state
    
    def _determine_action_node(self, state: DocumentState) -> DocumentState:
        """Node 4: Determine action based on document content with improved logic"""
        try:
            state["processing_steps"].append("⚖️ Step 4: Analyzing document for action keywords")
            
            text_lower = state["raw_text"].lower()
            
            # Define action patterns with negative keywords to avoid conflicts
            action_patterns = {
                "release_funds": {
                    "positive": ["release", "unfreeze", "restore", "return funds", "release funds", "unblock"],
                    "negative": ["do not release", "don't release", "not release"]
                },
                "freeze_funds": {
                    "positive": ["freeze", "frozen", "suspend funds", "block funds", "freeze funds"],
                    "negative": ["do not freeze", "don't freeze", "not freeze", "unfreeze", "release"]
                },
                "suspend_account": {
                    "positive": ["suspend account", "deactivate account", "suspend", "deactivate"],
                    "negative": ["do not suspend", "don't suspend", "not suspend", "activate", "reactivate"]
                },
                "transfer_funds": {
                    "positive": ["transfer", "move funds", "relocate funds", "transfer funds"],
                    "negative": ["do not transfer", "don't transfer", "not transfer"]
                }
            }
            
            action_scores = {}
            
            for action, patterns in action_patterns.items():
                positive_score = 0
                negative_score = 0
                
                # Count positive keywords
                for keyword in patterns["positive"]:
                    if keyword in text_lower:
                        positive_score += 1
                
                # Count negative keywords (these reduce the score)
                for neg_keyword in patterns["negative"]:
                    if neg_keyword in text_lower:
                        negative_score += 2  # Give more weight to negative keywords
                
                # Calculate final score (positive - negative)
                final_score = positive_score - negative_score
                
                if final_score > 0:
                    action_scores[action] = final_score
                    
            # Additional logic to handle "release" explicitly
            if "release" in text_lower and "freeze" in text_lower:
                # If both are present, check context
                if any(phrase in text_lower for phrase in ["do not freeze", "don't freeze", "not freeze"]):
                    # Explicitly favor release_funds
                    action_scores["release_funds"] = action_scores.get("release_funds", 0) + 3
                    # Remove or reduce freeze_funds score
                    if "freeze_funds" in action_scores:
                        del action_scores["freeze_funds"]
            
            if action_scores:
                # Get action with highest score
                determined_action = max(action_scores, key=action_scores.get)
                state["action"] = determined_action
                state["processing_steps"].append(f"✅ Action determined: {determined_action.upper()}")
                state["processing_steps"].append(f"📝 Action scores: {action_scores}")
                state["confidence"] *= 0.9
            else:
                state["action"] = None
                state["processing_steps"].append("⚠️ Action could not be determined from document")
                state["errors"].append("Unable to determine action from document content")
                state["confidence"] *= 0.4
                
        except Exception as e:
            state["errors"].append(f"Action determination failed: {str(e)}")
            state["confidence"] *= 0.3
            
        return state
    
    def _execute_action_node(self, state: DocumentState) -> DocumentState:
        """Node 5: Execute action for valid customer"""
        try:
            state["processing_steps"].append("🎯 Step 5: Executing action for validated customer")
            
            if state.get("action") and state.get("customer_info"):
                execution_result = self.action_processor.execute_action(
                    state["action"], 
                    state["customer_info"]
                )
                
                if execution_result["success"]:
                    state["processing_steps"].append(f"✅ {execution_result['message']}")
                    state["confidence"] *= 0.95
                else:
                    state["processing_steps"].append(f"❌ {execution_result['message']}")
                    state["errors"].append(execution_result["message"])
                    state["confidence"] *= 0.5
            else:
                state["processing_steps"].append("⚠️ Cannot execute action - missing action or customer info")
                state["errors"].append("Action execution skipped due to missing information")
                
        except Exception as e:
            state["errors"].append(f"Action execution failed: {str(e)}")
            state["confidence"] *= 0.3
            
        return state
    
    def _finalize_success_node(self, state: DocumentState) -> DocumentState:
        """Node 6a: Finalize successful processing"""
        try:
            state["processing_steps"].append("🎯 Step 6: Finalizing successful processing")
            state["workflow_completed"] = True
            state["rejected"] = False
            
            if state["national_id"] and state["action"] and state["customer_valid"]:
                state["processing_steps"].append("✅ Document processing completed successfully")
            else:
                state["processing_steps"].append("⚠️ Document processing completed with issues")
                
            state["processing_steps"].append(f"📊 Final confidence score: {state['confidence']:.2f}")
            
        except Exception as e:
            state["errors"].append(f"Finalization failed: {str(e)}")
            
        return state
    
    def _finalize_rejection_node(self, state: DocumentState) -> DocumentState:
        """Node 6b: Finalize rejected processing"""
        try:
            state["processing_steps"].append("🚫 Step 6: Finalizing document rejection")
            state["workflow_completed"] = True
            state["rejected"] = True
            
            rejection_msg = state.get("rejection_reason", "Document rejected due to validation failure")
            state["processing_steps"].append(f"❌ Document rejected: {rejection_msg}")
            state["processing_steps"].append("🔒 No actions executed - customer not found in database")
            
        except Exception as e:
            state["errors"].append(f"Rejection finalization failed: {str(e)}")
            
        return state
    
    def process_document(self, filename: str, text: str) -> Dict:
        """Process document using LangGraph workflow with customer validation"""
        
        # Initialize state
        initial_state: DocumentState = {
            "filename": filename,
            "raw_text": text,
            "national_id": None,
            "action": None,
            "confidence": 1.0,
            "processing_steps": [],
            "errors": [],
            "customer_valid": False,
            "customer_info": None,
            "workflow_completed": False,
            "rejected": False,
            "rejection_reason": None
        }
        
        # Run the LangGraph workflow
        final_state = self.graph.invoke(initial_state)
        
        # Return structured results
        return {
            "national_id": final_state.get("national_id"),
            "action": final_state.get("action"),
            "filename": final_state.get("filename"),
            "confidence": final_state.get("confidence", 0.0),
            "processing_steps": final_state.get("processing_steps", []),
            "errors": final_state.get("errors", []),
            "customer_valid": final_state.get("customer_valid", False),
            "customer_info": final_state.get("customer_info"),
            "rejected": final_state.get("rejected", False),
            "rejection_reason": final_state.get("rejection_reason"),
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
            "workflow_completed": final_state.get("workflow_completed", False)
        }

class DocumentExtractor:
    """Handles document text extraction"""
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF")

# Initialize components
customer_database = CustomerDatabase()
processing_agent = DocumentProcessingAgent(customer_database)
document_extractor = DocumentExtractor()

@app.post("/process_doc")
async def process_document(file: UploadFile = File(...)):
    """Process uploaded document using LangGraph agent with customer validation"""
    try:
        # Read file content
        content = await file.read()
        
        # Extract text
        if file.filename.lower().endswith('.pdf'):
            text = document_extractor.extract_text_from_pdf(content)
        else:
            text = content.decode('utf-8')
        
        # Process with LangGraph agent
        result = processing_agent.process_document(file.filename, text)
        
        logger.info(f"Processed document with customer validation: {file.filename}")
        
        return {
            "success": True,
            "data": result,
            "agent_workflow": "LangGraph processing with customer validation completed"
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_multiple")
async def process_multiple_documents(files: List[UploadFile] = File(...)):
    """Process multiple documents using LangGraph agent with customer validation"""
    try:
        results = []
        
        for file in files:
            content = await file.read()
            
            # Extract text
            if file.filename.lower().endswith('.pdf'):
                text = document_extractor.extract_text_from_pdf(content)
            else:
                text = content.decode('utf-8')
            
            # Process with LangGraph agent
            result = processing_agent.process_document(file.filename, text)
            results.append(result)
        
        return {
            "success": True,
            "data": results,
            "total_processed": len(results),
            "agent_framework": "LangGraph",
            "workflow_summary": "All documents processed through LangGraph workflow with customer validation"
        }
        
    except Exception as e:
        logger.error(f"Error processing multiple documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflow_info")
async def get_workflow_info():
    """Get information about the LangGraph workflow"""
    return {
        "framework": "LangGraph",
        "workflow_steps": [
            "1. Extract Text - Extract text from PDF or plain text files",
            "2. Identify National ID - Use regex patterns to find 10-digit IDs",
            "3. Validate Customer - Check if National ID exists in customer database",
            "4. Determine Action - Analyze keywords to classify actions (only for valid customers)",
            "5. Execute Action - Run dummy action functions for valid customers",
            "6. Finalize - Prepare final results or rejection notice"
        ],
        "node_types": ["extract_text", "identify_national_id", "validate_customer", "determine_action", "execute_action", "finalize_success", "finalize_rejection"],
        "supported_actions": ActionProcessor.SUPPORTED_ACTIONS,
        "customer_validation": True,
        "conditional_routing": "Documents are rejected if customer is not found in database"
    }

@app.get("/supported_actions")
async def get_supported_actions():
    """Get supported actions"""
    return {"supported_actions": ActionProcessor.SUPPORTED_ACTIONS}

@app.get("/customer_database")
async def get_customer_database():
    """Get customer database information"""
    return customer_database.get_database_info()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_framework": "LangGraph",
        "customer_database_loaded": customer_database.customers_df is not None,
        "total_customers": len(customer_database.customers_df) if customer_database.customers_df is not None else 0,
        "version": "1.0.0"
    }

# Test endpoint for debugging action determination
@app.post("/test_action")
async def test_action_determination(text: str):
    """Test action determination logic with provided text"""
    try:
        # Create a test document processing agent
        test_agent = DocumentProcessingAgent(customer_database)
        
        # Create test state
        test_state = {
            "filename": "test.txt",
            "raw_text": text,
            "national_id": None,
            "action": None,
            "confidence": 1.0,
            "processing_steps": [],
            "errors": [],
            "customer_valid": True,  # Bypass validation for testing
            "customer_info": {"customer_id": "TEST123", "national_id": "1234567890"},
            "workflow_completed": False,
            "rejected": False,
            "rejection_reason": None
        }
        
        # Run only the action determination node
        result_state = test_agent._determine_action_node(test_state)
        
        return {
            "input_text": text,
            "determined_action": result_state.get("action"),
            "processing_steps": result_state.get("processing_steps", []),
            "errors": result_state.get("errors", []),
            "confidence": result_state.get("confidence")
        }
        
    except Exception as e:
        logger.error(f"Error testing action determination: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)