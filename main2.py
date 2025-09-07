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

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Improved Court Document Processor with ML-Enhanced LangGraph")

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
    ml_confidence: Optional[float]
    fallback_method: Optional[str]

class ActionClassifier:
    """Improved Machine Learning-based action classifier with better negation handling"""
    
    def __init__(self, model_path="improved_action_classifier.pkl"):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 4),  # Increased to capture longer negation patterns
            max_features=10000,   # Increased features
            stop_words=None,     # Don't remove stop words as they're important for negation
            lowercase=True,
            token_pattern=r'\b\w+\b|DO_NOT|MUST_NOT|SHOULD_NOT|CANNOT|NOT_TO|NO_ACTION|NEVER|REFUSE_TO|PROHIBIT|FORBID'
        )
        self.classifier = MultinomialNB(alpha=0.01)  # Lower alpha for less smoothing
        self.model_path = model_path
        self.trained = False
        self.label_mapping = {
            'freeze_funds': 0,
            'release_funds': 1,
            'suspend_account': 2,
            'transfer_funds': 3
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
        # Try to load existing model
        self.load_model()
        
        # If no model exists, train with improved sample data
        if not self.trained:
            self.train_with_sample_data()
    
    def preprocess_text(self, text):
        """Enhanced preprocessing that handles negation better"""
        text = text.lower().strip()
        
        # Handle common negation patterns by adding explicit markers
        negation_patterns = [
            (r'\bdo not\b', ' DO_NOT '),
            (r'\bdon\'t\b', ' DO_NOT '),
            (r'\bmust not\b', ' MUST_NOT '),
            (r'\bmustn\'t\b', ' MUST_NOT '),
            (r'\bshould not\b', ' SHOULD_NOT '),
            (r'\bshouldn\'t\b', ' SHOULD_NOT '),
            (r'\bcannot\b', ' CANNOT '),
            (r'\bcan\'t\b', ' CANNOT '),
            (r'\bnot to\b', ' NOT_TO '),
            (r'\bno\s+', ' NO_ACTION '),
            (r'\bnever\b', ' NEVER '),
            (r'\brefuse to\b', ' REFUSE_TO '),
            (r'\bprohibit\b', ' PROHIBIT '),
            (r'\bforbid\b', ' FORBID ')
        ]
        
        for pattern, replacement in negation_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def generate_training_data(self):
        """Generate improved training data with better negation handling"""
        training_texts = [
            # Release funds examples (expanded with negation patterns)
            "The court orders the immediate release of all frozen funds for National ID 1234567890",
            "All previously frozen assets shall be restored to the account holder",
            "Unfreeze the account and allow normal transactions to resume",
            "The defendant's funds are to be released back to their account immediately",
            "Court orders the restoration of all blocked funds to National ID 9876543210",
            "Release all held funds and restore account access for the customer",
            "The freeze order is hereby lifted - restore full account functionality",
            "Unblock all transactions and return funds to the account holder",
            "Account restrictions are removed - release all frozen assets",
            "Funds shall be unfrozen and made available for withdrawal immediately",
            "Do not freeze this account, release the funds",
            "Do not keep the funds frozen, release them immediately",
            "Don't freeze the account, release all funds",
            "Must not freeze, should release the funds",
            "Cannot freeze this account, release the funds instead",
            "Not to freeze but to release all available funds",
            "Don't block the account, release all funds",
            "Should not freeze, must release the funds",
            "Court orders not to freeze but to release funds",
            "Do not maintain freeze, release all funds",
            "Never freeze this account, release the funds",
            "Refuse to freeze, release all funds immediately",
            "Prohibit freezing, allow fund release",
            "Forbid account freeze, release all funds",
            "No freezing allowed, release the funds",
            "Court says do not freeze, release funds",
            "Must not block account, release all funds",
            "Should not hold funds, release immediately",
            "Cannot keep frozen, must release funds",
            "Not permitted to freeze, release funds now",
            
            # Freeze funds examples
            "This court orders the immediate freeze of all funds in account linked to National ID 1234567890",
            "All assets belonging to the defendant must be frozen immediately due to court order",
            "The account associated with National ID 9876543210 shall be blocked from all transactions",
            "Funds must not be released and should be frozen until further notice",
            "This is to notify that account must not be released and to be frozen immediately due to legal orders",
            "Suspend all fund withdrawals for the account holder with National ID 1122334455",
            "Block all financial transactions for the defendant effective immediately",
            "The court hereby orders that all funds be frozen and not released until resolution",
            "Account funds are to be held and not released pending investigation",
            "Do not allow any fund releases - freeze the account immediately",
            "Must not release funds, freeze the account",
            "Should not unfreeze, keep the account frozen",
            "Cannot release funds, must freeze instead",
            "Not to release but to freeze all funds",
            "Don't unfreeze the account, keep it blocked",
            "Should freeze the account, not release funds",
            "Court orders to freeze, not to release",
            "Do not unblock the account, freeze all funds",
            "Never release funds, keep frozen",
            "Refuse to release, freeze immediately",
            "Prohibit fund release, freeze account",
            "Forbid unfreezing, maintain freeze status",
            
            # Suspend account examples
            "The court orders the suspension of account access for National ID 1234567890",
            "Account holder's access must be suspended but funds remain untouched",
            "Suspend the defendant's account login privileges temporarily",
            "Account access is to be deactivated pending further court review",
            "Temporarily suspend account operations for the customer",
            "The account should be suspended but not frozen",
            "Deactivate account access while maintaining fund balances",
            "Account privileges are hereby suspended until further notice",
            "Do not allow account access, suspend the account",
            "Don't permit login, suspend account access",
            "Must not allow access, suspend the account",
            "Cannot allow access, suspend immediately",
            "Not to permit access, suspend account",
            "Should not allow login, suspend user",
            "Never allow access, suspend account",
            "Refuse access, suspend immediately",
            
            # Transfer funds examples
            "The court orders the transfer of funds from National ID 1234567890 to court registry",
            "All available funds shall be transferred to the plaintiff's account",
            "Move all assets from defendant's account to escrow immediately",
            "Transfer the amount of $50000 from the account to court custody",
            "Funds are to be relocated from the defendant's account as per judgment",
            "Execute immediate transfer of all account balances to designated account",
            "Court directs the movement of funds to the settlement account",
            "Transfer all available assets from the account to the judgment creditor",
            "Do not keep funds in account, transfer them",
            "Don't leave funds there, transfer immediately",
            "Must not retain funds, transfer to court",
            "Cannot keep funds, must transfer",
            "Not to hold funds, transfer immediately",
            "Should not maintain funds, transfer now",
            "Never keep funds, transfer to court",
            "Refuse to hold funds, transfer immediately",
            
            # Complex negation examples with clear labels
            "The account linked to National ID 1234567890 must not be released but should be frozen",
            "Do not unfreeze the account - keep all funds blocked until court review",
            "The defendant's funds shall not be released and must remain frozen",
            "Account must not be unfrozen - maintain current freeze status",
            "Funds are not to be released - continue freeze until further orders",
            "The court finds that funds should not be released and orders freeze",
            "Do not release funds but freeze the account immediately",
            "Must not unblock the account - freeze all transactions",
            "Cannot unfreeze funds - keep them blocked",
            "Not permitted to release - freeze all assets"
        ]
        
        training_labels = [
            # Release funds labels (30 examples)
            'release_funds', 'release_funds', 'release_funds', 'release_funds', 'release_funds',
            'release_funds', 'release_funds', 'release_funds', 'release_funds', 'release_funds',
            'release_funds', 'release_funds', 'release_funds', 'release_funds', 'release_funds',
            'release_funds', 'release_funds', 'release_funds', 'release_funds', 'release_funds',
            'release_funds', 'release_funds', 'release_funds', 'release_funds', 'release_funds',
            'release_funds', 'release_funds', 'release_funds', 'release_funds', 'release_funds',
            
            # Freeze funds labels (22 examples)
            'freeze_funds', 'freeze_funds', 'freeze_funds', 'freeze_funds', 'freeze_funds',
            'freeze_funds', 'freeze_funds', 'freeze_funds', 'freeze_funds', 'freeze_funds',
            'freeze_funds', 'freeze_funds', 'freeze_funds', 'freeze_funds', 'freeze_funds',
            'freeze_funds', 'freeze_funds', 'freeze_funds', 'freeze_funds', 'freeze_funds',
            'freeze_funds', 'freeze_funds',
            
            # Suspend account labels (16 examples)
            'suspend_account', 'suspend_account', 'suspend_account', 'suspend_account',
            'suspend_account', 'suspend_account', 'suspend_account', 'suspend_account',
            'suspend_account', 'suspend_account', 'suspend_account', 'suspend_account',
            'suspend_account', 'suspend_account', 'suspend_account', 'suspend_account',
            
            # Transfer funds labels (16 examples)
            'transfer_funds', 'transfer_funds', 'transfer_funds', 'transfer_funds',
            'transfer_funds', 'transfer_funds', 'transfer_funds', 'transfer_funds',
            'transfer_funds', 'transfer_funds', 'transfer_funds', 'transfer_funds',
            'transfer_funds', 'transfer_funds', 'transfer_funds', 'transfer_funds',
            
            # Complex/negation examples - correct freeze classifications (10 examples)
            'freeze_funds', 'freeze_funds', 'freeze_funds', 'freeze_funds',
            'freeze_funds', 'freeze_funds', 'freeze_funds', 'freeze_funds',
            'freeze_funds', 'freeze_funds'
        ]
        
        return training_texts, training_labels
    
    def train_with_sample_data(self):
        """Train the classifier with improved sample data"""
        logger.info("Training improved ML classifier with enhanced court document data...")
        
        training_texts, training_labels = self.generate_training_data()
        self.train(training_texts, training_labels)
        
        # Save the trained model
        self.save_model()
        
        logger.info(f"Improved ML classifier trained with {len(training_texts)} samples")
    
    def train(self, training_texts, labels):
        """Train the classifier with provided texts and labels"""
        # Preprocess all training texts
        processed_texts = [self.preprocess_text(text) for text in training_texts]
        
        # Convert labels to numerical
        y = [self.label_mapping[label] for label in labels]
        
        # Fit vectorizer and transform texts
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Train classifier
        self.classifier.fit(X, y)
        self.trained = True
        
        # Calculate training accuracy
        y_pred = self.classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        logger.info(f"Training accuracy: {accuracy:.3f}")
        
        # Show class distribution
        unique, counts = np.unique(labels, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Class distribution: {class_dist}")
    
    def predict_action(self, text):
        """Predict action from text with improved preprocessing"""
        if not self.trained:
            logger.warning("Classifier not trained, cannot predict action")
            return None, 0.0
        
        # Preprocess the input text
        processed_text = self.preprocess_text(text)
        
        # Transform text
        X = self.vectorizer.transform([processed_text])
        
        # Get prediction and confidence
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = float(np.max(probabilities))
        
        # Convert numerical prediction back to label
        action = self.reverse_label_mapping[prediction]
        
        # Debug logging
        logger.info(f"Original text: {text}")
        logger.info(f"Processed text: {processed_text}")
        logger.info(f"Predicted action: {action} (confidence: {confidence:.3f})")
        
        # Show all probabilities for debugging
        all_probs = {self.reverse_label_mapping[i]: prob for i, prob in enumerate(probabilities)}
        logger.info(f"All probabilities: {all_probs}")
        
        return action, confidence
    
    def improved_fallback_method(self, text: str) -> Optional[str]:
        """Improved fallback method with better negation handling"""
        text_lower = text.lower()
        
        # Strong negation patterns for release funds (most specific first)
        release_negation_patterns = [
            r"do not freeze.*release",
            r"don\'t freeze.*release",
            r"must not freeze.*release",
            r"cannot freeze.*release",
            r"should not freeze.*release",
            r"not.*freeze.*release",
            r"never freeze.*release",
            r"refuse.*freeze.*release",
            r"prohibit.*freeze.*release",
            r"forbid.*freeze.*release",
        ]
        
        # Check for strong negation patterns first
        for pattern in release_negation_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"Found release negation pattern: {pattern}")
                return "release_funds"
        
        # Strong patterns for each action
        strong_patterns = {
            "release_funds": [
                r"release.*funds",
                r"unfreeze.*account",
                r"restore.*funds",
                r"lift.*freeze",
                r"unblock.*account",
                r"return.*funds",
                r"unfrozen.*funds",
                r"make.*available",
            ],
            
            "freeze_funds": [
                r"freeze.*funds",
                r"block.*account",
                r"must not.*release",
                r"not.*be.*released.*freeze",
                r"hold.*funds",
                r"restrict.*account",
                r"frozen.*immediately",
                r"block.*transactions",
            ],
            
            "suspend_account": [
                r"suspend.*account",
                r"deactivate.*access",
                r"block.*access",
                r"disable.*account",
                r"suspend.*login",
                r"deactivate.*privileges",
            ],
            
            "transfer_funds": [
                r"transfer.*funds",
                r"move.*assets",
                r"relocate.*funds",
                r"send.*funds",
                r"transfer.*amount",
                r"move.*money",
            ]
        }
        
        scores = {}
        for action, pattern_list in strong_patterns.items():
            score = sum(1 for pattern in pattern_list if re.search(pattern, text_lower))
            if score > 0:
                scores[action] = score
        
        # If we found matches, return the highest scoring action
        if scores:
            best_action = max(scores, key=scores.get)
            logger.info(f"Fallback method scores: {scores}, chosen: {best_action}")
            return best_action
        
        logger.info("No patterns matched in fallback method")
        return None
    
    def save_model(self):
        """Save the trained model to disk"""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_mapping': self.label_mapping,
            'reverse_label_mapping': self.reverse_label_mapping,
            'trained': self.trained
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Improved ML model saved to {self.model_path}")
    
    def load_model(self):
        """Load trained model from disk"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.vectorizer = model_data['vectorizer']
                self.classifier = model_data['classifier']
                self.label_mapping = model_data['label_mapping']
                self.reverse_label_mapping = model_data['reverse_label_mapping']
                self.trained = model_data['trained']
                
                logger.info(f"Improved ML model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                self.trained = False
        else:
            logger.info("No pre-trained model found, will train new improved model")

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
        # Add some default values for display purposes
        customer_data['name'] = f"Customer {customer_data.get('customer_id', 'Unknown')}"
        customer_data['account_status'] = 'active'  # Default status
        customer_data['account_balance'] = 0.00  # Default balance
        
        return {
            "valid": True,
            "customer_info": customer_data,
            "reason": "Customer found and validated"
        }
    
    def get_database_info(self) -> Dict:
        """Get database information"""
        if self.customers_df is None:
            return {"total_customers": 0, "customers": []}
        
        # Convert to records and add default display values
        customers_list = []
        for _, row in self.customers_df.iterrows():
            customer_record = row.to_dict()
            customer_record['name'] = f"Customer {customer_record.get('customer_id', 'Unknown')}"
            customer_record['account_status'] = 'active'
            customer_record['account_balance'] = 0.00
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
            customer_name = customer_info.get('name')
            
            if action == "freeze_funds":
                return {
                    "success": True,
                    "message": f"✅ Funds frozen for customer {customer_name} (ID: {national_id})",
                    "action_details": "All account funds have been frozen per court order"
                }
            elif action == "release_funds":
                return {
                    "success": True,
                    "message": f"✅ Funds released for customer {customer_name} (ID: {national_id})",
                    "action_details": "All frozen funds have been released back to customer account"
                }
            elif action == "suspend_account":
                return {
                    "success": True,
                    "message": f"✅ Account suspended for customer {customer_name} (ID: {national_id})",
                    "action_details": "Customer account has been temporarily suspended"
                }
            elif action == "transfer_funds":
                return {
                    "success": True,
                    "message": f"✅ Fund transfer initiated for customer {customer_name} (ID: {national_id})",
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
    """LangGraph-based document processing agent with improved ML-enhanced action detection"""
    
    def __init__(self, customer_db: CustomerDatabase):
        self.customer_db = customer_db
        self.action_processor = ActionProcessor()
        self.action_classifier = ActionClassifier()  # Improved ML classifier
        self.graph = self._build_processing_graph()
        self.id_pattern = r'\b\d{10}\b'
        
    def _build_processing_graph(self) -> StateGraph:
        """Build the LangGraph processing workflow with improved ML-enhanced action detection"""
        
        # Create the graph
        workflow = StateGraph(DocumentState)
        
        # Add nodes (processing steps)
        workflow.add_node("extract_text", self._extract_text_node)
        workflow.add_node("identify_national_id", self._identify_national_id_node)
        workflow.add_node("validate_customer", self._validate_customer_node)
        workflow.add_node("determine_action_ml", self._determine_action_ml_node)
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
                "process": "determine_action_ml",
                "reject": "finalize_rejection"
            }
        )
        
        workflow.add_edge("determine_action_ml", "execute_action")
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
                customer_name = state["customer_info"].get("name", "Unknown")
                state["processing_steps"].append(f"✅ Customer validated: {customer_name}")
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
    
    def _determine_action_ml_node(self, state: DocumentState) -> DocumentState:
        """Node 4: Determine action using improved ML classifier with better fallback"""
        try:
            state["processing_steps"].append("🤖 Step 4: Using improved ML classifier to determine action")
            
            # Primary method: Improved ML classification
            ml_action, ml_confidence = self.action_classifier.predict_action(state["raw_text"])
            state["ml_confidence"] = ml_confidence
            
            if ml_action and ml_confidence > 0.4:  # Lowered threshold for improved model
                state["action"] = ml_action
                state["fallback_method"] = "ML_Primary"
                state["processing_steps"].append(f"✅ ML Action determined: {ml_action.upper()} (confidence: {ml_confidence:.2f})")
                state["confidence"] *= ml_confidence
            else:
                # Fallback: Improved keyword matching
                state["processing_steps"].append(f"⚠️ ML confidence too low ({ml_confidence:.2f}), using improved fallback method")
                fallback_action = self.action_classifier.improved_fallback_method(state["raw_text"])
                
                if fallback_action:
                    state["action"] = fallback_action
                    state["fallback_method"] = "Improved_Keyword_Fallback"
                    state["processing_steps"].append(f"✅ Fallback action determined: {fallback_action.upper()}")
                    state["confidence"] *= 0.8  # Higher confidence for improved fallback
                else:
                    state["action"] = ml_action  # Use ML prediction even if low confidence
                    state["fallback_method"] = "ML_LowConfidence"
                    state["processing_steps"].append(f"⚠️ Using ML prediction with low confidence: {ml_action}")
                    state["confidence"] *= ml_confidence
            
        except Exception as e:
            state["errors"].append(f"ML Action determination failed: {str(e)}")
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
                    method = state.get("fallback_method", "Unknown")
                    state["processing_steps"].append(f"✅ {execution_result['message']} (Method: {method})")
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
                if state.get("ml_confidence"):
                    state["processing_steps"].append(f"🤖 ML confidence: {state['ml_confidence']:.2f}")
                if state.get("fallback_method"):
                    state["processing_steps"].append(f"🔧 Method used: {state['fallback_method']}")
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
        """Process document using LangGraph workflow with improved ML-enhanced action detection"""
        
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
            "rejection_reason": None,
            "ml_confidence": None,
            "fallback_method": None
        }
        
        # Run the LangGraph workflow
        final_state = self.graph.invoke(initial_state)
        
        # Return structured results
        return {
            "national_id": final_state.get("national_id"),
            "action": final_state.get("action"),
            "filename": final_state.get("filename"),
            "confidence": final_state.get("confidence", 0.0),
            "ml_confidence": final_state.get("ml_confidence"),
            "fallback_method": final_state.get("fallback_method"),
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
    """Process uploaded document using improved ML-enhanced LangGraph agent"""
    try:
        # Read file content
        content = await file.read()
        
        # Extract text
        if file.filename.lower().endswith('.pdf'):
            text = document_extractor.extract_text_from_pdf(content)
        else:
            text = content.decode('utf-8')
        
        # Process with improved ML-enhanced LangGraph agent
        result = processing_agent.process_document(file.filename, text)
        
        logger.info(f"Processed document with improved ML-enhanced validation: {file.filename}")
        
        return {
            "success": True,
            "data": result,
            "agent_workflow": "Improved ML-Enhanced LangGraph processing with customer validation completed",
            "ml_enabled": True,
            "improvements": "Enhanced negation handling and better training data"
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_text")
async def process_text_directly(text_data: dict):
    """Process text directly without file upload (useful for testing)"""
    try:
        text = text_data.get("text", "")
        filename = text_data.get("filename", "direct_text_input.txt")
        
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        # Process with improved ML-enhanced LangGraph agent
        result = processing_agent.process_document(filename, text)
        
        logger.info(f"Processed text directly: {text[:50]}...")
        
        return {
            "success": True,
            "data": result,
            "agent_workflow": "Improved ML-Enhanced LangGraph direct text processing completed",
            "ml_enabled": True,
            "improvements": "Enhanced negation handling and better training data"
        }
        
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_multiple")
async def process_multiple_documents(files: List[UploadFile] = File(...)):
    """Process multiple documents using improved ML-enhanced LangGraph agent"""
    try:
        results = []
        
        for file in files:
            content = await file.read()
            
            # Extract text
            if file.filename.lower().endswith('.pdf'):
                text = document_extractor.extract_text_from_pdf(content)
            else:
                text = content.decode('utf-8')
            
            # Process with improved ML-enhanced LangGraph agent
            result = processing_agent.process_document(file.filename, text)
            results.append(result)
        
        return {
            "success": True,
            "data": results,
            "total_processed": len(results),
            "agent_framework": "Improved ML-Enhanced LangGraph",
            "workflow_summary": "All documents processed through improved ML-enhanced LangGraph workflow with customer validation",
            "ml_enabled": True,
            "improvements": "Enhanced negation handling and better training data"
        }
        
    except Exception as e:
        logger.error(f"Error processing multiple documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflow_info")
async def get_workflow_info():
    """Get information about the improved ML-enhanced LangGraph workflow"""
    return {
        "framework": "Improved ML-Enhanced LangGraph",
        "version": "2.0.0-Improved",
        "workflow_steps": [
            "1. Extract Text - Extract text from PDF or plain text files",
            "2. Identify National ID - Use regex patterns to find 10-digit IDs",
            "3. Validate Customer - Check if National ID exists in customer database",
            "4. Improved ML Action Detection - Use enhanced classifier with better negation handling",
            "5. Execute Action - Run dummy action functions for valid customers",
            "6. Finalize - Prepare final results or rejection notice"
        ],
        "node_types": ["extract_text", "identify_national_id", "validate_customer", "determine_action_ml", "execute_action", "finalize_success", "finalize_rejection"],
        "supported_actions": ActionProcessor.SUPPORTED_ACTIONS,
        "customer_validation": True,
        "ml_enhanced": True,
        "ml_confidence_threshold": 0.4,
        "improvements": {
            "negation_handling": "Enhanced preprocessing with explicit negation markers",
            "training_data": "94 samples with balanced negation examples",
            "fallback_method": "Improved keyword matching with negation patterns",
            "vectorizer": "Enhanced with custom token patterns for negation markers",
            "model_tuning": "Lower alpha parameter for less smoothing"
        },
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

@app.get("/ml_model_info")
async def get_ml_model_info():
    """Get improved ML model information"""
    return {
        "model_type": "Multinomial Naive Bayes (Improved)",
        "vectorizer": "TF-IDF with enhanced n-grams (1-4) and negation tokens",
        "trained": processing_agent.action_classifier.trained,
        "confidence_threshold": 0.4,
        "supported_actions": list(processing_agent.action_classifier.label_mapping.keys()),
        "training_samples": "94 synthetic court documents with enhanced negation handling",
        "improvements": {
            "negation_preprocessing": "Explicit negation markers (DO_NOT, MUST_NOT, etc.)",
            "balanced_training": "30 release, 22 freeze, 16 suspend, 16 transfer, 10 complex negation",
            "enhanced_patterns": "Better regex patterns for negation detection",
            "lower_smoothing": "Alpha=0.01 for better feature sensitivity"
        },
        "fallback_method": "Improved keyword matching with negation detection"
    }

@app.post("/retrain_model")
async def retrain_ml_model():
    """Retrain the improved ML model with fresh data"""
    try:
        processing_agent.action_classifier.train_with_sample_data()
        return {
            "success": True,
            "message": "Improved ML model retrained successfully",
            "model_info": {
                "trained": processing_agent.action_classifier.trained,
                "model_path": processing_agent.action_classifier.model_path,
                "improvements": "Enhanced negation handling and balanced training data"
            }
        }
    except Exception as e:
        logger.error(f"Error retraining improved model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrain improved model: {str(e)}")

@app.get("/test_negation")
async def test_negation_examples():
    """Test the improved model with various negation examples"""
    test_cases = [
        "1234567890 Do not freeze this account, release the funds",
        "1234567890 Don't freeze the account, release all funds",
        "1234567890 Must not freeze, should release the funds",
        "1234567890 Cannot freeze this account, release the funds instead",
        "1234567890 Never freeze this account, release the funds",
        "1234567890 Freeze this account immediately",
        "1234567890 Release all funds from this account",
        "1234567890 Do not release funds, freeze the account",
        "1234567890 Must not be released and should be frozen",
        "1234567890 Transfer all funds to court registry"
    ]
    
    results = []
    for text in test_cases:
        try:
            # Test ML classifier
            ml_action, ml_confidence = processing_agent.action_classifier.predict_action(text)
            
            # Test fallback method
            fallback_action = processing_agent.action_classifier.improved_fallback_method(text)
            
            results.append({
                "text": text,
                "ml_prediction": ml_action,
                "ml_confidence": round(ml_confidence, 3),
                "fallback_prediction": fallback_action,
                "expected": "Based on text content"
            })
        except Exception as e:
            results.append({
                "text": text,
                "error": str(e)
            })
    
    return {
        "test_results": results,
        "model_version": "Improved with negation handling"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_framework": "Improved ML-Enhanced LangGraph",
        "customer_database_loaded": customer_database.customers_df is not None,
        "total_customers": len(customer_database.customers_df) if customer_database.customers_df is not None else 0,
        "ml_model_trained": processing_agent.action_classifier.trained,
        "ml_model_path": processing_agent.action_classifier.model_path,
        "version": "2.0.0-Improved",
        "improvements": {
            "negation_handling": True,
            "enhanced_training_data": True,
            "improved_fallback": True,
            "better_preprocessing": True
        }
    }

@app.get("/")
async def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Improved Court Document Processor with ML-Enhanced LangGraph",
        "version": "2.0.0-Improved",
        "features": [
            "Enhanced negation handling",
            "Improved ML classification",
            "Better training data",
            "Advanced fallback methods",
            "Customer database validation",
            "LangGraph workflow"
        ],
        "endpoints": {
            "POST /process_doc": "Process uploaded document",
            "POST /process_text": "Process text directly",
            "POST /process_multiple": "Process multiple documents",
            "GET /workflow_info": "Get workflow information",
            "GET /ml_model_info": "Get ML model details",
            "GET /test_negation": "Test negation handling",
            "POST /retrain_model": "Retrain ML model",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)