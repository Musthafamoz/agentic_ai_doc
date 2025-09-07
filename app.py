import streamlit as st
import requests
import json
from typing import List, Dict
import time
import pandas as pd
from io import BytesIO

# Configure Streamlit page
st.set_page_config(
    page_title="Court Document Processor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .success-card {
        border-left-color: #27ae60;
    }
    
    .error-card {
        border-left-color: #e74c3c;
    }
    
    .rejected-card {
        border-left-color: #f39c12;
        background: #fdf2e9;
    }
    
    .workflow-step {
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        background: #f8f9fa;
    }
    
    .step-success {
        background: #d4edda;
        color: #155724;
    }
    
    .step-error {
        background: #f8d7da;
        color: #721c24;
    }
    
    .step-warning {
        background: #fff3cd;
        color: #856404;
    }
    
    .step-rejection {
        background: #fdf2e9;
        color: #d68910;
    }
</style>
""", unsafe_allow_html=True)

# Backend API configuration
BACKEND_URL = "http://localhost:8000"

class DocumentProcessor:
    """Handle communication with FastAPI backend"""
    
    @staticmethod
    def process_single_document(file) -> Dict:
        """Process a single document via API"""
        try:
            files = {"file": (file.name, file.getvalue(), file.type)}
            response = requests.post(f"{BACKEND_URL}/process_doc", files=files)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to backend API. Make sure the FastAPI server is running on port 8000.")
            return None
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            return None
    
    @staticmethod
    def process_multiple_documents(files: List) -> Dict:
        """Process multiple documents via API"""
        try:
            file_data = [("files", (f.name, f.getvalue(), f.type)) for f in files]
            response = requests.post(f"{BACKEND_URL}/process_multiple", files=file_data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to backend API. Make sure the FastAPI server is running on port 8000.")
            return None
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
            return None
    
    @staticmethod
    def get_workflow_info() -> Dict:
        """Get workflow information from API"""
        try:
            response = requests.get(f"{BACKEND_URL}/workflow_info")
            response.raise_for_status()
            return response.json()
        except:
            return None
    
    @staticmethod
    def health_check() -> bool:
        """Check if backend API is healthy"""
        try:
            response = requests.get(f"{BACKEND_URL}/health")
            return response.status_code == 200
        except:
            return False

def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>Document Processor</h1>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display sidebar with information and controls"""
    with st.sidebar:
        st.header("🔧 System Info")
        
        # Health check
        if DocumentProcessor.health_check():
            st.success("✅ Backend API Connected")
        else:
            st.error("❌ Backend API Disconnected")
            st.info("Make sure FastAPI server is running:\n```bash\npython main.py```")
        
        st.header("📋 Supported Actions")
        # Get supported actions from backend
        try:
            response = requests.get(f"{BACKEND_URL}/supported_actions")
            if response.status_code == 200:
                actions = response.json().get("supported_actions", {})
                for action, description in actions.items():
                    st.info(f"**{action}**: {description}")
            else:
                st.info("""
                **freeze_funds**: Freeze all funds in the customer account
                
                **release_funds**: Release all held or restricted funds back to the customer account
                """)
        except:
            st.info("""
            **freeze_funds**: Freeze all funds in the customer account
            
            **release_funds**: Release all held or restricted funds back to the customer account
            """)
        
        st.header("👥 Customer Database")
        # Show customer database info
        try:
            response = requests.get(f"{BACKEND_URL}/customer_database")
            if response.status_code == 200:
                db_info = response.json()
                st.success(f"✅ {db_info['total_customers']} customers loaded")
                
                with st.expander("View Customer Database", expanded=False):
                    customers_df = pd.DataFrame(db_info['customers'])
                    st.dataframe(customers_df, use_container_width=True)
                    
                    # Show sample National IDs for testing
                    st.subheader("🔍 Sample National IDs for Testing")
                    sample_ids = customers_df['national_id'].head(3).tolist()
                    for nid in sample_ids:
                        st.code(nid, language=None)
                    st.caption("Use these National IDs in your test documents")
            else:
                st.warning("Could not load customer database")
        except:
            st.warning("Could not connect to customer database")
        
        # Workflow information
        workflow_info = DocumentProcessor.get_workflow_info()
        if workflow_info:
            st.header("📄 LangGraph Workflow")
            for i, step in enumerate(workflow_info.get("workflow_steps", []), 1):
                st.text(f"{i}. {step.split(' - ')[1] if ' - ' in step else step}")
        
        st.header("📄 Sample Documents")
        st.info("""
        Upload court orders to test the system:
        - Documents must contain 10-digit National IDs
        - Only customers in the database will be processed
        - Actions determined by keyword analysis
        - Non-customers are automatically rejected
        """)

def display_confidence_bar(confidence: float):
    """Display confidence score as a progress bar"""
    if confidence >= 0.8:
        color = "green"
    elif confidence >= 0.5:
        color = "orange"
    else:
        color = "red"
    
    st.progress(confidence)
    st.caption(f"Confidence: {confidence:.1%} ({color.upper()})")

def display_processing_steps(steps: List[str], errors: List[str] = None):
    """Display LangGraph processing steps"""
    st.subheader("📄 LangGraph Processing Steps")
    
    for step in steps:
        if "✅" in step:
            st.markdown(f'<div class="workflow-step step-success">{step}</div>', unsafe_allow_html=True)
        elif "❌" in step:
            st.markdown(f'<div class="workflow-step step-error">{step}</div>', unsafe_allow_html=True)
        elif "⚠️" in step:
            st.markdown(f'<div class="workflow-step step-warning">{step}</div>', unsafe_allow_html=True)
        elif "🚫" in step:
            st.markdown(f'<div class="workflow-step step-rejection">{step}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="workflow-step">{step}</div>', unsafe_allow_html=True)
    
    if errors:
        st.subheader("⚠️ Processing Errors")
        for error in errors:
            st.error(f"• {error}")

# def display_customer_info(customer_info: Dict):
#     """Display customer information"""
#     if customer_info:
#         st.subheader("👤 Customer Information")
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.info(f"**Name**: {customer_info.get('name', 'N/A')}")
#             st.info(f"**National ID**: {customer_info.get('national_id', 'N/A')}")
        
#         with col2:
#             st.info(f"**Account Status**: {customer_info.get('account_status', 'N/A')}")
#             balance = customer_info.get('account_balance', 0)
#             st.info(f"**Account Balance**: ${balance:,.2f}")

def display_single_result(result: Dict):
    """Display results for a single document"""
    # Check if document was rejected
    if result.get('rejected'):
        st.error(f"🚫 Document Rejected: {result['filename']}")
        st.warning(f"**Reason**: {result.get('rejection_reason', 'Unknown reason')}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if result.get('national_id'):
                st.error(f"**National ID Found**: {result['national_id']} (Not in customer database)")
            else:
                st.error("**National ID**: Not found")
        
        with col2:
            st.subheader("📊 Confidence Score")
            confidence = result.get('confidence', 0.0)
            display_confidence_bar(confidence)
    
    else:
        # Document was processed successfully
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📄 {result['filename']}")
            
            # Main results
            subcol1, subcol2, subcol3 = st.columns(3)
            
            with subcol1:
                if result.get('national_id'):
                    if result.get('customer_valid'):
                        st.success(f"**National ID**: {result['national_id']} ✅")
                    else:
                        st.error(f"**National ID**: {result['national_id']} ❌")
                else:
                    st.error("**National ID**: Not found")
            
            with subcol2:
                if result.get('action'):
                    action_color = "🟢" if result['action'] == 'release_funds' else "🔴"
                    st.info(f"**Action**: {action_color} {result['action']}")
                else:
                    st.error("**Action**: Not determined")
            
            with subcol3:
                if result.get('workflow_completed'):
                    if result.get('customer_valid'):
                        st.success("**Status**: ✅ Processed")
                    else:
                        st.error("**Status**: ❌ Rejected")
                else:
                    st.error("**Status**: ❌ Failed")
        
        with col2:
            st.subheader("📊 Confidence Score")
            confidence = result.get('confidence', 0.0)
            display_confidence_bar(confidence)
        
        # Customer information
        # if result.get('customer_valid') and result.get('customer_info'):
        #     display_customer_info(result['customer_info'])
    
    # Processing steps
    if result.get('processing_steps'):
        with st.expander("🔍 View LangGraph Processing Details", expanded=False):
            display_processing_steps(result['processing_steps'], result.get('errors'))
    
    # Document preview
    if result.get('text_preview'):
        with st.expander("📖 Document Preview", expanded=False):
            st.text_area("Document Content", result['text_preview'], height=150, disabled=True)

def display_batch_results(results: List[Dict]):
    """Display results for multiple documents"""
    # Summary statistics
    total_docs = len(results)
    successful_extractions = sum(1 for r in results if r.get('national_id') and r.get('action') and r.get('customer_valid'))
    rejected_docs = sum(1 for r in results if r.get('rejected'))
    valid_customers = sum(1 for r in results if r.get('customer_valid'))
    freeze_count = sum(1 for r in results if r.get('action') == 'freeze_funds' and r.get('customer_valid'))
    release_count = sum(1 for r in results if r.get('action') == 'release_funds' and r.get('customer_valid'))
    avg_confidence = sum(r.get('confidence', 0) for r in results) / total_docs if total_docs > 0 else 0
    
    st.header("📊 Processing Summary")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Documents", total_docs)
    
    with col2:
        st.metric("Processed", successful_extractions, f"{successful_extractions/total_docs:.1%}" if total_docs > 0 else "0%")
    
    with col3:
        st.metric("Rejected", rejected_docs, f"🚫 {rejected_docs}")
    
    with col4:
        st.metric("Valid Customers", valid_customers, f"👥 {valid_customers}")
    
    # with col5:
    #     st.metric("Freeze Orders", freeze_count, f"🔴 {freeze_count}")
    
    # with col6:
    #     st.metric("Release Orders", release_count, f"🟢 {release_count}")
    
    # # Additional metrics row
    # col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     st.metric("Avg Confidence", f"{avg_confidence:.1%}", f"{'📈' if avg_confidence > 0.7 else '📉'}")
    
    # with col2:
    #     success_rate = (valid_customers / total_docs) * 100 if total_docs > 0 else 0
    #     st.metric("Success Rate", f"{success_rate:.1%}", f"{'✅' if success_rate > 50 else '⚠️'}")
    
    # with col3:
    #     rejection_rate = (rejected_docs / total_docs) * 100 if total_docs > 0 else 0
    #     st.metric("Rejection Rate", f"{rejection_rate:.1%}", f"{'🚫' if rejection_rate > 0 else '✅'}")
    
    # Results table
    st.header("📋 Detailed Results")
    
    # Create DataFrame for better visualization
    df_data = []
    for result in results:
        status = "🚫 Rejected" if result.get('rejected') else ("✅ Processed" if result.get('customer_valid') else "❌ Failed")
        
        df_data.append({
            'Filename': result['filename'],
            'National ID': result.get('national_id', 'Not found'),
            'Customer Valid': '✅ Yes' if result.get('customer_valid') else '❌ No',
            'Action': result.get('action', 'Unknown'),
            'Confidence': f"{result.get('confidence', 0):.1%}",
            'Status': status
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # Individual results
    st.header("📄 Individual Results")
    for i, result in enumerate(results):
        status_icon = "🚫" if result.get('rejected') else ("✅" if result.get('customer_valid') else "❌")
        with st.expander(f"{status_icon} {result['filename']}", expanded=False):
            display_single_result(result)

def main():
    """Main Streamlit application"""
    display_header()
    display_sidebar()
    
    # Main content area
    st.header("📤 Upload Documents")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose court document files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF or text files containing court orders with National IDs"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) uploaded: {', '.join([f.name for f in uploaded_files])}")
        
        # Processing options
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            process_single = st.button("🔍 Process Single", disabled=len(uploaded_files) != 1)
        
        with col2:
            process_all = st.button("⚡ Process All", disabled=len(uploaded_files) == 0)
        
        with col3:
            if st.button("🗑️ Clear All"):
                st.rerun()
        
        # Process single document
        if process_single and len(uploaded_files) == 1:
            with st.spinner("📄 Processing document with customer validation..."):
                result = DocumentProcessor.process_single_document(uploaded_files[0])
                
                if result and result.get('success'):
                    if result['data'].get('rejected'):
                        st.warning("⚠️ Document processed but rejected due to customer validation failure")
                    else:
                        st.success("✅ Document processed successfully!")
                    display_single_result(result['data'])
                else:
                    st.error("❌ Failed to process document")
        
        # Process all documents
        if process_all:
            with st.spinner(f"📄 Processing {len(uploaded_files)} documents with customer validation..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate progress (since we're doing batch processing)
                for i in range(len(uploaded_files)):
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    status_text.text(f"Processing {uploaded_files[i].name}...")
                    time.sleep(0.2)  # Small delay for visual effect
                
                result = DocumentProcessor.process_multiple_documents(uploaded_files)
                
                progress_bar.empty()
                status_text.empty()
                
                if result and result.get('success'):
                    st.success(f"✅ Successfully processed {len(result['data'])} documents!")
                    display_batch_results(result['data'])
                else:
                    st.error("❌ Failed to process documents")
    
    else:
        st.info("""
        👆 **Upload court documents to get started**
        
        📋 **Instructions:**
        1. Upload PDF or text files containing court orders
        2. Choose to process single document or all at once
        3. System validates National IDs against customer database
        4. View extracted information and processing results
        5. Check LangGraph workflow steps and confidence scores
        
        ⚠️ **Important Notes:**
        - Only processes documents for registered customers
        - Court orders for unknown customers are automatically rejected
        - Actions are executed only for valid customer documents
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🤖 <strong>Court Document Processor</strong> | Powered by LangGraph & FastAPI | Built with Streamlit</p>
        <p>⚖️ Intelligent document processing with customer validation for legal workflows</p>
        <p>🔒 Secure processing - only handles documents for registered customers</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()