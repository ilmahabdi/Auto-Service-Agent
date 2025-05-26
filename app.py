import logging
import os
from datetime import datetime, timedelta
import streamlit as st
from pydantic import BaseModel, Field, validator, constr
from typing import Literal, Optional, Dict, Any, List, Union
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import AzureChatOpenAI
from composio import ComposioToolSet
import pandas as pd
import json
import time
import uuid
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
# --- Configuration and Constants
SERVICE_CENTERS = {
    "Premier Service Center": {"address": "123 Main St, Downtown", "phone": "555-123-4567", "capacity": 15},
    "Elite Auto Hub": {"address": "456 Park Ave, Uptown", "phone": "555-765-4321", "capacity": 12},
    "Precision Auto Care": {"address": "789 Oak Rd, Westside", "phone": "555-987-6543", "capacity": 18}
}

TIME_SLOTS = [
    "08:00-10:00", "10:00-12:00", "12:00-14:00",
    "14:00-16:00", "16:00-18:00"
]

SERVICE_TYPES = {
    "Standard Maintenance": 199.99,
    "Oil Change": 49.99,
    "Tire Rotation": 39.99,
    "Brake Service": 179.99,
    "Full Inspection": 149.99,
    "Engine Diagnostics": 129.99,
    "A/C Service": 109.99,
    "Battery Replacement": 89.99,
    "Transmission Service": 219.99
}

VEHICLE_MANUFACTURERS = [
    "Acura", "Audi", "BMW", "Buick", "Cadillac", "Chevrolet", "Chrysler", "Dodge", "Ford", "GMC",
    "Honda", "Hyundai", "Infiniti", "Jaguar", "Jeep", "Kia", "Land Rover", "Lexus", "Lincoln",
    "Mazda", "Mercedes-Benz", "Mitsubishi", "Nissan", "Porsche", "Ram", "Subaru", "Tesla",
    "Toyota", "Volkswagen", "Volvo"
]

SERVICE_INTERVALS = {
    "Oil Change": {"miles": 5000, "months": 6},
    "Tire Rotation": {"miles": 7500, "months": 6},
    "Brake Inspection": {"miles": 12000, "months": 12},
    "Air Filter": {"miles": 15000, "months": 12},
    "Transmission Service": {"miles": 30000, "months": 24},
    "Spark Plugs": {"miles": 60000, "months": 36},
    "Timing Belt": {"miles": 90000, "months": 60}
}

# --- Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AutoXConnect")

# --- Page Configuration
st.set_page_config(
    page_title="AutoX Connect",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Apply custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .block-container {
        padding: 2rem 1rem;
    }
    .main-header {
        font-family: 'Arial', sans-serif;
        color: #1e3a8a;
        text-align: center;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .service-header {
        color: #2563eb;
        font-weight: 600;
    }
    .service-subtitle {
        color: #64748b;
    }
    .metric-container {
        background-color: #f0f9ff;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #0369a1;
    }
    .metric-label {
        font-size: 14px;
        color: #64748b;
    }
    .tab-switcher {
        display: flex;
        overflow: hidden;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .tab-button {
        background-color: #e2e8f0;
        border: none;
        outline: none;
        cursor: pointer;
        padding: 14px 16px;
        transition: 0.3s;
        font-size: 16px;
        flex: 1;
        text-align: center;
    }
    .tab-button.active {
        background-color: #2563eb;
        color: white;
    }
    .progress-tracker {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .progress-step {
        flex: 1;
        text-align: center;
        position: relative;
    }
    .progress-step:after {
        content: '';
        position: absolute;
        width: 100%;
        height: 3px;
        background-color: #e2e8f0;
        top: 15px;
        left: 50%;
        z-index: -1;
    }
    .progress-step:last-child:after {
        display: none;
    }
    .progress-step-circle {
        display: inline-block;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background-color: #e2e8f0;
        color: #64748b;
        text-align: center;
        line-height: 30px;
        margin-bottom: 5px;
    }
    .progress-step.active .progress-step-circle {
        background-color: #2563eb;
        color: white;
    }
    .progress-step.completed .progress-step-circle {
        background-color: #10b981;
        color: white;
    }
    .progress-step-label {
        font-size: 12px;
        color: #64748b;
    }
    .progress-step.active .progress-step-label {
        color: #2563eb;
        font-weight: bold;
    }
    .alert-info {
        padding: 15px;
        border-radius: 8px;
        background-color: #e0f2fe;
        border-left: 4px solid #0ea5e9;
        margin-bottom: 20px;
    }
    .alert-success {
        padding: 15px;
        border-radius: 8px;
        background-color: #dcfce7;
        border-left: 4px solid #10b981;
        margin-bottom: 20px;
    }
    .alert-warning {
        padding: 15px;
        border-radius: 8px;
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        margin-bottom: 20px;
    }
    .alert-danger {
        padding: 15px;
        border-radius: 8px;
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8fafc;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #bfdbfe !important;
        color: #1e40af;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for workflow management
if 'page' not in st.session_state:
    st.session_state.page = "vehicle_check"
if 'customer_id' not in st.session_state:
    st.session_state.customer_id = str(uuid.uuid4())
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'service_history' not in st.session_state:
    st.session_state.service_history = []
if 'fleet_data' not in st.session_state:
    # Sample fleet data for enterprise view
    st.session_state.fleet_data = [
        {"id": "FLT-001", "make": "Toyota", "model": "Camry", "year": 2021, "last_service": "2024-11-15", "status": "Active", "department": "Sales"},
        {"id": "FLT-002", "make": "Honda", "model": "Accord", "year": 2022, "last_service": "2025-01-20", "status": "Active", "department": "Operations"},
        {"id": "FLT-003", "make": "Ford", "model": "F-150", "year": 2020, "last_service": "2025-02-05", "status": "Maintenance", "department": "Delivery"},
        {"id": "FLT-004", "make": "Chevrolet", "model": "Equinox", "year": 2023, "last_service": "2025-03-10", "status": "Active", "department": "Executive"},
        {"id": "FLT-005", "make": "Nissan", "model": "Altima", "year": 2021, "last_service": "2024-12-18", "status": "Inactive", "department": "Sales"}
    ]

# --- Header with Logo and Title
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://via.placeholder.com/100x100", width=80)
with col2:
    st.markdown("<h1 class='main-header'>AutoX Connect</h1>", unsafe_allow_html=True)
    st.markdown("<p class='service-subtitle'>Enterprise Automotive Service Management Platform</p>", unsafe_allow_html=True)

# --- Sidebar Configuration
with st.sidebar:
    st.markdown("## 🧭 Navigation")

    # Role selector for different views
    user_role = st.selectbox(
        "Select User Role",
        ["Service Advisor", "Service Manager", "Fleet Manager", "System Administrator"],
        help="Change view based on your role"
    )

    # Navigation based on role
    if user_role == "Service Advisor":
        nav_option = st.radio(
            "Navigate to:",
            ["Vehicle Service Check", "Customer Management", "Appointment Calendar", "Parts Inventory"]
        )
    elif user_role == "Service Manager":
        nav_option = st.radio(
            "Navigate to:",
            ["Service Dashboard", "Technician Scheduling", "Performance Reports", "Customer Feedback"]
        )
    elif user_role == "Fleet Manager":
        nav_option = st.radio(
            "Navigate to:",
            ["Fleet Overview", "Maintenance Schedule", "Cost Analytics", "Vehicle History"]
        )
    else:  # System Administrator
        nav_option = st.radio(
            "Navigate to:",
            ["System Configuration", "User Management", "Integration Settings", "Audit Logs"]
        )

    st.markdown("---")

    st.markdown("## ⚙️ Configuration")

    with st.expander("🔑 Azure OpenAI Settings", expanded=False):
        endpoint = st.text_input(
            "Azure OpenAI Endpoint",
            value=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            placeholder="https://<your-resource>.openai.azure.com/",
            help="Your Azure OpenAI service endpoint URL"
        )
        api_key = st.text_input(
            "API Key",
            value=os.environ.get("AZURE_OPENAI_KEY", ""),
            type="password",
            help="Your Azure OpenAI API key"
        )
        api_ver = st.text_input(
            "API Version",
            value=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            help="Azure OpenAI API version to use"
        )
        deployment = st.text_input(
            "Model Deployment",
            value=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-1"),
            help="Your Azure OpenAI model deployment name"
        )

    with st.expander("🔌 Integration Settings", expanded=False):
        composio_key = st.text_input(
            "Composio API Key",
            value=os.environ.get("COMPOSIO_API_KEY", ""),
            type="password",
            help="API key for Composio integration services"
        )
        integration_id = st.text_input(
            "Calendar Integration ID",
            value=os.environ.get("CALENDAR_INTEGRATION_ID", "55aca871-c683-44fa-b9e6-ab24267890ae"),
            help="Composio calendar integration identifier"
        )

        # Additional integrations
        st.checkbox("Enable DMS Integration", value=True, help="Connect with Dealer Management System")
        st.checkbox("Enable Parts Catalog", value=True, help="Connect with parts inventory system")
        st.checkbox("Enable SMS Notifications", value=True, help="Send automated SMS notifications")
        st.checkbox("Enable Payment Processing", value=False, help="Enable integrated payment processing")

    # --- Help section
    st.markdown("---")
    st.markdown("## 📋 Quick Links")
    st.markdown("""
    - [Service Guidelines](https://example.com)
    - [Technical Documentation](https://example.com)
    - [Training Videos](https://example.com)
    - [Company Portal](https://example.com)
    """)

    # --- Support contact
    st.markdown("---")
    st.markdown("## 🛟 Technical Support")
    st.markdown("Need assistance? Contact IT support at support@autox.com")
    st.markdown("System Version: 2.4.7 (May 2025)")

    # Additional tools for certain roles
    if user_role in ["Service Manager", "System Administrator"]:
        st.markdown("---")
        st.markdown("## 🛠️ Admin Tools")
        st.button("Sync Database", key="sync_db")
        st.button("Clear Cache", key="clear_cache")

# --- Data models for structured parsing
class ServiceDue(BaseModel):
    service_due: Literal["yes", "no"] = Field(...)
    reason: str = Field(...)
    recommended_services: List[str] = Field(...)
    urgency_level: Literal["low", "medium", "high"] = Field(...)
    estimated_costs: Dict[str, float] = Field(...)

class UserPreferences(BaseModel):
    preferred_date: constr(pattern=r"^\d{4}-\d{2}-\d{2}$") = Field(...)
    time_slot: str
    service_center: str
    service_type: str
    customer_notes: Optional[str] = None
    shuttle_service: Optional[bool] = False
    loaner_vehicle: Optional[bool] = False

    @validator('service_center')
    def validate_service_center(cls, v):
        if v not in SERVICE_CENTERS:
            raise ValueError(f"Invalid service center. Must be one of {list(SERVICE_CENTERS.keys())}")
        return v

    @validator('time_slot')
    def validate_time_slot(cls, v):
        if v not in TIME_SLOTS:
            raise ValueError(f"Invalid time slot. Must be one of {TIME_SLOTS}")
        return v

    @validator('service_type')
    def validate_service_type(cls, v):
        if v not in SERVICE_TYPES:
            raise ValueError(f"Invalid service type. Must be one of {list(SERVICE_TYPES.keys())}")
        return v

class Vehicle(BaseModel):
    make: str
    model: str
    year: int
    registration: str
    vin: str
    last_service_date: constr(pattern=r"^\d{4}-\d{2}-\d{2}$") = Field(...)
    mileage: int
    fuel_type: Literal["Gasoline", "Diesel", "Electric", "Hybrid", "Plug-in Hybrid"] = "Gasoline"
    engine_size: Optional[str] = None
    transmission: Literal["Automatic", "Manual", "CVT", "DCT"] = "Automatic"
    warranty_status: Optional[bool] = True

    @validator('year')
    def validate_year(cls, v):
        current_year = datetime.now().year
        if v < 1900 or v > current_year + 1:
            raise ValueError(f"Invalid year, must be between 1900 and {current_year + 1}")
        return v

    @validator('mileage')
    def validate_mileage(cls, v):
        if v < 0 or v > 1000000:
            raise ValueError("Mileage must be between 0 and 1,000,000")
        return v

    @validator('vin')
    def validate_vin(cls, v):
        if len(v) != 17:
            raise ValueError("VIN must be 17 characters long")
        return v

class Customer(BaseModel):
    id: str
    name: str
    email: str
    phone: str
    address: Optional[str] = None
    preferred_communication: Literal["Email", "SMS", "Phone"] = "Email"
    loyalty_tier: Literal["Standard", "Silver", "Gold", "Platinum"] = "Standard"
    vehicles: List[str] = []  # List of vehicle VINs

# --- Initialize Azure OpenAI if credentials are provided
def init_llm():
    if not (endpoint and api_key and api_ver and deployment):
        return None

    try:
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_ver,
            deployment_name=deployment,
            temperature=0.0
        )
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI: {e}")
        return None

# --- Initialize Composio if credentials are provided
def init_composio():
    if not (composio_key and integration_id):
        return None

    try:
        toolset = ComposioToolSet(api_key=composio_key)
        return toolset.get_integration(id=integration_id)
    except Exception as e:
        logger.error(f"Failed to initialize Composio: {e}")
        return None

# --- Build LLM chains
def build_chains(llm):
    # Templates with more details and professional language
    detect_parser = PydanticOutputParser(pydantic_object=ServiceDue)


    detect_template = PromptTemplate(
         template="""
      You are an expert automotive service advisor with access to manufacturer-specific service intervals and recommendations.

   {format_instructions}

  Vehicle details:
- Make/Model: {make} {model}
   - Year: {year}
   - VIN: {vin}
   - Last service date: {last_service_date}
   - Current mileage: {mileage}
   - Registration: {registration}
   - Fuel Type: {fuel_type}
   - Transmission: {transmission}
   - Warranty Status: {warranty_status}

   Based on these details, determine if the vehicle is due for service considering:
   1. Time-based service intervals for this specific make and model
   2. Mileage-based service intervals according to manufacturer recommendations
   3. Seasonal maintenance requirements based on current date
   4. Warranty requirements to maintain coverage

   Include specific service recommendations based on:
   - Each system (engine, transmission, brakes, etc.)
   - Safety-critical components
   - Warranty requirements
   - Recall information if available
   """,
       input_variables=[
           "make", "model", "year", "vin", "last_service_date",
           "mileage", "registration", "fuel_type", "transmission", "warranty_status"
       ],
       partial_variables={"format_instructions": detect_parser.get_format_instructions()}
   )


    book_template = PromptTemplate.from_template(
    """
    You are a professional automotive service advisor at a premium automotive service center.

    The customer's {year} {make} {model} requires service for the following reason:
    {reason}

    Recommended services based on their vehicle status:
    {recommended_services}

    Urgency level: {urgency_level}

    Provide a premium customer experience by:
    1. Creating a polite and professional message informing the customer of the required services.
    2. Emphasizing the urgency level using customer-friendly language.
    3. Encouraging them to book an appointment soon to ensure vehicle safety and maintain warranty coverage.
    4. Offering reassurance that certified technicians will handle their vehicle with care.

    Keep the message concise, informative, and warm in tone. Respond in a paragraph suitable for sending via email or chat.
    """
    )

    collect_template = PromptTemplate.from_template(
        """
        You are a service booking assistant collecting appointment preferences.
        The customer has agreed to book a service appointment.

        Available service centers:
        {service_centers}

        Available time slots:
        {time_slots}

        Available service types:
        {service_types}

        Ask the customer for their preferences in a professional, conversational manner:
        - Preferred date (YYYY-MM-DD)
        - Time slot (from the available options)
        - Service center location (from the available options)
        - Type of service needed (from the available options)
        - Any additional notes for the service team

        Format your response as a friendly, professional service representative.
        """
    )
    collect_parser = PydanticOutputParser(pydantic_object=UserPreferences)

    # Build chains
    detect_chain = LLMChain(
        llm=llm,
        prompt=detect_template,
        output_key="service_assessment"
    )

    book_chain = LLMChain(
        llm=llm,
        prompt=book_template,
        output_key="booking_prompt"
    )

    collect_chain = LLMChain(
        llm=llm,
        prompt=collect_template,
        output_key="user_prefs",
        output_parser=collect_parser
    )

    return detect_chain, book_chain, collect_chain

# --- Create appointment in calendar
def create_calendar_event(cal_integration, prefs, vehicle):
    try:
        start_time, end_time = prefs.time_slot.split("-")
        start = datetime.fromisoformat(f"{prefs.preferred_date}T{start_time}:00")
        end = datetime.fromisoformat(f"{prefs.preferred_date}T{end_time}:00")

        location_info = SERVICE_CENTERS[prefs.service_center]

        event = {
            "summary": f"{vehicle.make} {vehicle.model} - {prefs.service_type}",
            "description": (
                f"Vehicle: {vehicle.year} {vehicle.make} {vehicle.model}\n"
                f"Registration: {vehicle.registration}\n"
                f"Current Mileage: {vehicle.mileage}\n"
                f"Service Type: {prefs.service_type}\n"
                f"Customer Notes: {prefs.customer_notes or 'None provided'}"
            ),
            "start": {"dateTime": start.isoformat(), "timeZone": "Asia/Kolkata"},
            "end": {"dateTime": end.isoformat(), "timeZone": "Asia/Kolkata"},
            "location": f"{prefs.service_center}: {location_info['address']}"
        }

        return cal_integration.google_calendar.create_event(event)
    except Exception as e:
        logger.error(f"Failed to create calendar event: {e}")
        raise e

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

def ask_the_agent_tab():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='service-header'>🔍 Ask the AutoX Agent</h3>", unsafe_allow_html=True)

    llm = init_llm()
    if not llm:
        st.error("⚠️ Please configure Azure OpenAI credentials.")
        return

    # Memory setup
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Data filter options
    data_filter = st.radio("Select context to use:", ["Fleet Data", "Service History", "Both"], horizontal=True)

    # User query
    user_input = st.text_input("Ask a question about your fleet or service data:")

    if st.button("Ask") and user_input:
        # Prepare context
        fleet_data = st.session_state.get("fleet_data", [])
        service_history = st.session_state.get("service_history", [])

        if data_filter == "Fleet Data":
           context = f"Fleet Data:\n{json.dumps(fleet_data, indent=2)}" 
        elif data_filter == "Service History":
           context = f"Service History:\n{json.dumps(service_history, indent=2)}"
        else:
           context = (
               f"Fleet Data:\n{json.dumps(fleet_data, indent=2)}\n\n"
               f"Service History:\n{json.dumps(service_history, indent=2)}"
           )


        # Inject context using partial
        prompt_template = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", """Context:
{context}

{input}""")
        ]).partial(context=context)

        # Conversation chain with memory
        conversation = ConversationChain(
            llm=llm,
            prompt=prompt_template,
            memory=st.session_state.chat_memory,
            verbose=False
        )

        # Get LLM response
        with st.spinner("Thinking..."):
            response = conversation.run(user_input)
            st.success(response)

    # Show conversation history
    if st.session_state.chat_memory.buffer_as_messages:
        st.markdown("### 💬 Chat History")
        for msg in st.session_state.chat_memory.buffer_as_messages:
            if msg.type == "human":
                st.markdown(f"**You:** {msg.content}")
            elif msg.type == "ai":
                st.markdown(f"**Agent:** {msg.content}")

    st.markdown("</div>", unsafe_allow_html=True)


# --- Main application tabs
tab1, tab2, tab3 = st.tabs(["Vehicle Service Check", "Service History", "Ask the Agent"])

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='service-header'>Vehicle Details</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        make = st.text_input("Make", value="Toyota", help="Vehicle manufacturer")
        model = st.text_input("Model", value="Camry", help="Vehicle model")
        year = st.number_input("Year", min_value=1950, max_value=datetime.now().year + 1, value=2020, help="Year of manufacture")

    with col2:
        registration = st.text_input("Registration Number", value="ABC-123", help="Vehicle registration/license plate")
        vin = st.text_input("Vehicle VIN", value="1HGCM82633A123456", help="17-character vehicle identification number")
        last_service_date = st.date_input(
            "Last Service Date",
            datetime.now() - timedelta(days=180),
            help="Date of the last service performed"
        )
        mileage = st.number_input(
            "Current Mileage",
            min_value=0,
            max_value=1000000,
            value=15000,
            help="Current odometer reading"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Check service eligibility button
    check_service = st.button("Check Service Status", type="primary", use_container_width=True)

    if check_service:
        # Initialize Azure OpenAI
        llm = init_llm()
        if not llm:
            st.error("⚠️ Azure OpenAI credentials are missing or invalid. Please check your settings.")
            st.stop()

        # Show loading spinner
        with st.spinner("Analyzing service status..."):
            try:
                # Create vehicle object
                vehicle = Vehicle(
                    make=make,
                    model=model,
                    year=year,
                    registration=registration,
                    vin=vin,
                    last_service_date=last_service_date.isoformat(),
                    mileage=mileage
                )

                # Build and run detect chain
                detect_chain, book_chain, collect_chain = build_chains(llm)

                # Get raw response from LLM
                raw_service_result = detect_chain.run(
                    make=vehicle.make,
                    model=vehicle.model,
                    year=vehicle.year,
                    vin=vehicle.vin,
                    last_service_date=vehicle.last_service_date,
                    mileage=vehicle.mileage,
                    registration=vehicle.registration,
                    fuel_type=vehicle.fuel_type,
                    transmission=vehicle.transmission,
                    warranty_status=vehicle.warranty_status
                )

                # Parse the raw JSON response
                try:
                    # Try to directly parse the response
                    service_result = json.loads(raw_service_result)
                except json.JSONDecodeError:
                    # If not valid JSON, try to extract JSON from the text
                    import re
                    json_match = re.search(r'(\{.*\})', raw_service_result, re.DOTALL)
                    if json_match:
                        service_result = json.loads(json_match.group(1))
                    else:
                        raise ValueError("Could not extract valid JSON from LLM response")

                # Create ServiceDue object from the parsed result
                service_assessment = ServiceDue(**service_result)

                # Display service assessment
                st.markdown("<div class='card'>", unsafe_allow_html=True)

                if service_assessment.service_due == "yes":
                    st.markdown("<h3 style='color:#e11d48'>Service Required ⚠️</h3>", unsafe_allow_html=True)
                    st.write(f"**Reason:** {service_assessment.reason}")

                    st.markdown("### Recommended Services:")
                    for service in service_assessment.recommended_services:
                        st.markdown(f"- {service}")

                    # Show estimated costs
                    st.markdown("### Estimated Costs:")
                    for service, cost in service_assessment.estimated_costs.items():
                        st.markdown(f"- {service}: ${cost:.2f}")

                    # Show urgency level with appropriate styling
                    urgency_color = {
                        "low": "green",
                        "medium": "orange",
                        "high": "red"
                    }.get(service_assessment.urgency_level, "gray")

                    st.markdown(f"**Urgency Level:** <span style='color:{urgency_color};font-weight:bold;'>{service_assessment.urgency_level.upper()}</span>", unsafe_allow_html=True)

                    # Run booking prompt chain
                    booking_message = book_chain.run(
                        year=vehicle.year,
                        make=vehicle.make,
                        model=vehicle.model,
                        reason=service_assessment.reason,
                        recommended_services=", ".join(service_assessment.recommended_services),
                        urgency_level=service_assessment.urgency_level
                    )

                    st.info(booking_message)

                    # Ask if user wants to schedule now
                    # Ask if user wants to schedule now
                    schedule_service = st.button("Schedule Service Now", type="primary")

                    if schedule_service:
                        st.session_state.page = "booking"
                        st.session_state.vehicle = vehicle
                        st.session_state.service_assessment = service_assessment
                        st.experimental_rerun()

                else:
                    st.markdown("<h3 style='color:#10b981'>No Service Required ✓</h3>", unsafe_allow_html=True)
                    st.write(f"**Analysis:** {service_assessment.reason}")

                    if service_assessment.recommended_services:
                        st.markdown("### Preventative Maintenance Suggested:")
                        for service in service_assessment.recommended_services:
                            st.markdown(f"- {service}")

                    st.success("Your vehicle is in good condition. We recommend checking back in 3 months or 3,000 miles, whichever comes first.")

                st.markdown("</div>", unsafe_allow_html=True)

                # Store service history
                history_entry = {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "vehicle": f"{vehicle.year} {vehicle.make} {vehicle.model}",
                    "registration": vehicle.registration,
                    "assessment": "Service Required" if service_assessment.service_due == "yes" else "No Service Required",
                    "recommendations": service_assessment.recommended_services,
                    "urgency": service_assessment.urgency_level
                }
                st.session_state.service_history.append(history_entry)

            except Exception as e:
                st.error(f"An error occurred during service assessment: {str(e)}")
                logger.error(f"Service assessment error: {e}")

# --- Service booking workflow if in booking page
if st.session_state.page == "booking":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='service-header'>Schedule Service Appointment</h3>", unsafe_allow_html=True)

    # Progress tracker
    st.markdown("<div class='progress-tracker'>", unsafe_allow_html=True)
    for step, label in enumerate([
        "Vehicle Check",
        "Select Services",
        "Choose Date & Time",
        "Confirm Details",
        "Appointment Booked"
    ], 1):
        status = ""
        if step < st.session_state.current_step:
            status = "completed"
        elif step == st.session_state.current_step:
            status = "active"

        st.markdown(f"""
        <div class='progress-step {status}'>
            <div class='progress-step-circle'>{step}</div>
            <div class='progress-step-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    vehicle = st.session_state.vehicle
    service_assessment = st.session_state.service_assessment

    # Step 2: Select Services
    if st.session_state.current_step == 2:
        st.markdown("### Select Services")
        st.markdown("Please select the services you would like to include:")

        selected_services = {}
        total_cost = 0.0

        # Display recommended services with checkboxes
        for service in service_assessment.recommended_services:
            cost = service_assessment.estimated_costs.get(service, 0.0)
            selected = st.checkbox(f"{service} (${cost:.2f})", value=True)
            if selected:
                selected_services[service] = cost
                total_cost += cost

        # Additional services
        st.markdown("### Additional Services")
        for service, cost in SERVICE_TYPES.items():
            if service not in service_assessment.recommended_services:
                selected = st.checkbox(f"{service} (${cost:.2f})")
                if selected:
                    selected_services[service] = cost
                    total_cost += cost

        # Display total
        st.markdown(f"### Total Estimated Cost: ${total_cost:.2f}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", key="back_to_step1"):
                st.session_state.current_step = 1
                st.session_state.page = "vehicle_check"
                st.experimental_rerun()

        with col2:
            if st.button("Next", key="proceed_to_step3", type="primary"):
                st.session_state.selected_services = selected_services
                st.session_state.total_cost = total_cost
                st.session_state.current_step = 3
                st.experimental_rerun()

    # Step 3: Choose Date & Time
    elif st.session_state.current_step == 3:
        st.markdown("### Select Date & Service Center")

        col1, col2 = st.columns(2)

        with col1:
            # Date picker for appointment date
            min_date = datetime.now().date() + timedelta(days=1)
            max_date = datetime.now().date() + timedelta(days=30)
            selected_date = st.date_input(
                "Preferred Date",
                min_value=min_date,
                max_value=max_date,
                value=min_date + timedelta(days=3),
                help="Select your preferred service date"
            )

            # Service center selection
            service_center = st.selectbox(
                "Service Center",
                list(SERVICE_CENTERS.keys()),
                help="Choose your preferred service location"
            )

            # Display service center details
            center_details = SERVICE_CENTERS[service_center]
            st.markdown(f"""
            **Address:** {center_details['address']}
            **Phone:** {center_details['phone']}
            **Capacity:** {center_details['capacity']} vehicles per day
            """)

        with col2:
            # Time slot selection with availability
            st.markdown("### Select Time Slot")
            selected_slot = None

            for slot in TIME_SLOTS:
                # Simulate availability - in a real app, this would check against a database
                import random
                available = random.choice([True, True, True, False])
                disabled = not available

                if st.button(
                    f"{slot} {'(Available)' if available else '(Booked)'}",
                    disabled=disabled,
                    key=f"slot_{slot}",
                    use_container_width=True
                ):
                    selected_slot = slot

            st.markdown("### Additional Options")
            shuttle_service = st.checkbox("Need shuttle service", value=False)
            loaner_vehicle = st.checkbox("Request loaner vehicle", value=False)
            customer_notes = st.text_area("Notes for service team", height=100)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", key="back_to_step2"):
                st.session_state.current_step = 2
                st.experimental_rerun()

        with col2:
            proceed_button = st.button("Next", key="proceed_to_step4", type="primary")
            if proceed_button:
                if not selected_slot:
                    st.error("Please select a time slot to continue")
                else:
                    # Store appointment preferences
                    st.session_state.appointment_prefs = {
                        "date": selected_date.isoformat(),
                        "time_slot": selected_slot,
                        "service_center": service_center,
                        "shuttle_service": shuttle_service,
                        "loaner_vehicle": loaner_vehicle,
                        "customer_notes": customer_notes
                    }
                    st.session_state.current_step = 4
                    st.experimental_rerun()

    # Step 4: Confirm Details
    elif st.session_state.current_step == 4:
        st.markdown("### Confirm Appointment Details")

        # Create columns for appointment summary
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Vehicle Information")
            st.markdown(f"""
            **Vehicle:** {vehicle.year} {vehicle.make} {vehicle.model}
            **Registration:** {vehicle.registration}
            **VIN:** {vehicle.vin}
            **Current Mileage:** {vehicle.mileage}
            """)

            st.markdown("#### Appointment Details")
            prefs = st.session_state.appointment_prefs
            st.markdown(f"""
            **Date:** {prefs['date']}
            **Time:** {prefs['time_slot']}
            **Service Center:** {prefs['service_center']}
            **Address:** {SERVICE_CENTERS[prefs['service_center']]['address']}
            """)

            if prefs['shuttle_service']:
                st.markdown("**Shuttle Service:** Requested")
            if prefs['loaner_vehicle']:
                st.markdown("**Loaner Vehicle:** Requested")
            if prefs['customer_notes']:
                st.markdown(f"**Notes:** {prefs['customer_notes']}")

        with col2:
            st.markdown("#### Selected Services")
            services = st.session_state.selected_services
            for service, cost in services.items():
                st.markdown(f"- {service}: ${cost:.2f}")

            st.markdown(f"#### Total Cost: ${st.session_state.total_cost:.2f}")

            # Payment method selection
            st.markdown("#### Payment Information")
            payment_method = st.selectbox(
                "Payment Method",
                ["Credit Card", "Debit Card", "Invoice to Company", "Insurance Coverage"]
            )

            contact_pref = st.radio(
                "Contact Preference",
                ["Email", "SMS", "Phone Call"]
            )

        st.markdown("### Terms and Conditions")
        terms_agreed = st.checkbox("I agree to the terms and conditions for vehicle servicing", value=False)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back", key="back_to_step3"):
                st.session_state.current_step = 3
                st.experimental_rerun()

        with col2:
            confirm_button = st.button("Confirm Booking", key="confirm_booking", type="primary", disabled=not terms_agreed)
            if confirm_button:
                with st.spinner("Processing your booking..."):
                    try:
                        # Initialize calendar integration
                        cal_integration = init_composio()

                        if cal_integration:
                            # Create user preferences object
                            user_prefs = UserPreferences(
                                preferred_date=prefs['date'],
                                time_slot=prefs['time_slot'],
                                service_center=prefs['service_center'],
                                service_type=list(services.keys())[0],  # Primary service
                                customer_notes=prefs['customer_notes'],
                                shuttle_service=prefs['shuttle_service'],
                                loaner_vehicle=prefs['loaner_vehicle']
                            )

                            # Create calendar event
                            event_result = create_calendar_event(cal_integration, user_prefs, vehicle)

                            # Store booking reference
                            st.session_state.booking_reference = f"SVC-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
                            st.session_state.current_step = 5
                            st.experimental_rerun()
                        else:
                            # Simulate successful booking for demo purposes
                            time.sleep(2)
                            st.session_state.booking_reference = f"SVC-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
                            st.session_state.current_step = 5
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Booking failed: {str(e)}")
                        logger.error(f"Booking error: {e}")

    # Step 5: Booking Confirmation
    elif st.session_state.current_step == 5:
        st.markdown("<div class='alert-success'>", unsafe_allow_html=True)
        st.markdown("### 🎉 Appointment Successfully Booked!")
        st.markdown(f"**Booking Reference:** {st.session_state.booking_reference}")
        st.markdown("</div>", unsafe_allow_html=True)

        prefs = st.session_state.appointment_prefs

        st.markdown("### Appointment Details")
        st.markdown(f"""
        Your vehicle service appointment has been confirmed for **{prefs['date']}** at **{prefs['time_slot']}**.

        Please arrive at **{prefs['service_center']}** located at **{SERVICE_CENTERS[prefs['service_center']]['address']}** about 10 minutes before your appointment time.

        A confirmation email has been sent to your registered email address with all the details.
        """)

        st.markdown("### What's Next?")
        st.markdown("""
        - You'll receive a reminder 24 hours before your appointment
        - Our service advisor will greet you upon arrival
        - You'll receive updates while your vehicle is being serviced
        - We'll notify you when your vehicle is ready for pickup
        """)

        if st.button("Return to Vehicle Check", type="primary"):
            st.session_state.page = "vehicle_check"
            st.session_state.current_step = 1
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# Service History Tab
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='service-header'>Service History</h3>", unsafe_allow_html=True)

    if not st.session_state.service_history:
        st.info("No service history available yet. Complete a vehicle check to start building history.")
    else:
        # Display history table
        history_df = pd.DataFrame(st.session_state.service_history)
        st.dataframe(history_df, use_container_width=True)

        # Create metrics
        st.markdown("### Service Analytics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            checks_count = len(history_df)
            st.markdown(f"<div class='metric-value'>{checks_count}</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Total Vehicle Checks</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            service_required = len(history_df[history_df["assessment"] == "Service Required"])
            percentage = int((service_required / checks_count) * 100) if checks_count > 0 else 0
            st.markdown(f"<div class='metric-value'>{percentage}%</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Service Required Rate</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with tab3:
             ask_the_agent_tab()
        with col3:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            unique_vehicles = history_df["vehicle"].nunique()
            st.markdown(f"<div class='metric-value'>{unique_vehicles}</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Unique Vehicles</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Visualizations
        st.markdown("### Service Trends")

        # Create a sample chart
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Service Recommendations", "Urgency Levels"),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )

        # Extract recommendations for visualization
        all_recommendations = []
        for recs in history_df["recommendations"]:
            all_recommendations.extend(recs)

        from collections import Counter
        recommendation_counts = Counter(all_recommendations)

        # Bar chart for recommendations
        fig.add_trace(
            go.Bar(
                x=list(recommendation_counts.keys())[:5],  # Top 5 recommendations
                y=list(recommendation_counts.values())[:5],
                marker_color='#3b82f6'
            ),
            row=1, col=1
        )

        # Pie chart for urgency levels
        urgency_counts = history_df["urgency"].value_counts()

        fig.add_trace(
            go.Pie(
                labels=urgency_counts.index,
                values=urgency_counts.values,
                marker=dict(colors=['#22c55e', '#f59e0b', '#ef4444'])
            ),
            row=1, col=2
        )

        fig.update_layout(height=400, width=800)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --- Fleet Manager Dashboard
if user_role == "Fleet Manager" and nav_option == "Fleet Overview":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='service-header'>Fleet Management Dashboard</h3>", unsafe_allow_html=True)

    # Fleet metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value'>5</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Vehicles</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value'>4</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Active Vehicles</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value'>1</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Vehicles Needing Service</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value'>$12,480</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Maintenance Budget</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Fleet table
    st.markdown("### Fleet Inventory")
    fleet_df = pd.DataFrame(st.session_state.fleet_data)
    st.dataframe(fleet_df, use_container_width=True)

    # Fleet visualizations
    st.markdown("### Fleet Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Vehicle status pie chart
        status_counts = fleet_df["status"].value_counts()
        fig1 = px.pie(
            names=status_counts.index,
            values=status_counts.values,
            title="Vehicle Status Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Department allocation bar chart
        dept_counts = fleet_df["department"].value_counts()
        fig2 = px.bar(
            x=dept_counts.index,
            y=dept_counts.values,
            title="Vehicles by Department",
            labels={"x": "Department", "y": "Number of Vehicles"},
            color_discrete_sequence=['#3b82f6']
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Maintenance calendar
    st.markdown("### Upcoming Maintenance Schedule")

    # Sample upcoming maintenance data
    upcoming_maintenance = [
        {"vehicle_id": "FLT-001", "vehicle": "Toyota Camry", "service_type": "Oil Change", "due_date": "2025-05-20", "assigned": "Elite Auto Hub"},
        {"vehicle_id": "FLT-003", "vehicle": "Ford F-150", "service_type": "Full Inspection", "due_date": "2025-05-25", "assigned": "Premier Service Center"},
        {"vehicle_id": "FLT-004", "vehicle": "Chevrolet Equinox", "service_type": "Tire Rotation", "due_date": "2025-06-10", "assigned": "Precision Auto Care"}
    ]

    maint_df = pd.DataFrame(upcoming_maintenance)
    st.dataframe(maint_df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --- Service Manager Dashboard
if user_role == "Service Manager" and nav_option == "Service Dashboard":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='service-header'>Service Operations Dashboard</h3>", unsafe_allow_html=True)

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.now() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.now()
        )

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value'>87</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Appointments</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value'>92%</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Completion Rate</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value'>4.8</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Customer Rating</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value'>$34,250</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Revenue</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Performance charts
    st.markdown("### Service Center Performance")

    # Generate sample data for a month
    dates = pd.date_range(start=start_date, end=end_date)

    service_data = {
        "date": dates,
        "appointments": [random.randint(5, 15) for _ in range(len(dates))],
        "completion_rate": [random.uniform(0.85, 1.0) for _ in range(len(dates))],
        "avg_service_time": [random.uniform(1.5, 4.5) for _ in range(len(dates))]
    }

    service_df = pd.DataFrame(service_data)

    # Create performance chart
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=service_df["date"],
            y=service_df["appointments"],
            name="Appointments",
            line=dict(color="#3b82f6", width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=service_df["date"],
            y=service_df["completion_rate"] * 10,  # Scale for visibility
            name="Completion Rate (× 10)",
            line=dict(color="#10b981", width=2, dash="dot")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=service_df["date"],
            y=service_df["avg_service_time"],
            name="Avg Service Time (hours)",
            line=dict(color="#f59e0b", width=2)
        )
    )

    fig.update_layout(
        title="Daily Performance Metrics",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Service center comparison
    st.markdown("### Service Center Comparison")

    centers = list(SERVICE_CENTERS.keys())
    metrics = ["Appointments", "Revenue", "Customer Rating", "Efficiency"]

    comparison_data = {
        center: [random.randint(70, 100) for _ in range(len(metrics))]
        for center in centers
    }

    radar_fig = go.Figure()

    for center, values in comparison_data.items():
        radar_fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=center
        ))

    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        )
    )

    st.plotly_chart(radar_fig, use_container_width=True)

    # Technician performance
    st.markdown("### Technician Performance")

    technicians = ["John Smith", "Maria Garcia", "David Chen", "Fatima Ali", "Robert Johnson"]

    tech_data = {
        "Technician": technicians,
        "Jobs Completed": [random.randint(15, 30) for _ in range(len(technicians))],
        "Average Time (hours)": [random.uniform(1.8, 3.2) for _ in range(len(technicians))],
        "Customer Rating": [random.uniform(4.0, 5.0) for _ in range(len(technicians))],
        "Efficiency Score": [random.uniform(80, 98) for _ in range(len(technicians))]
    }

    tech_df = pd.DataFrame(tech_data)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.dataframe(tech_df, use_container_width=True)

    with col2:
        # Efficiency score gauge chart
        efficiency_fig = go.Figure()

        for i, tech in enumerate(technicians):
            efficiency_fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=tech_df.loc[i, "Efficiency Score"],
                domain={'x': [0, 1], 'y': [i/len(technicians), (i+0.8)/len(technicians)]},
                title={'text': tech},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#3b82f6"},
                    'steps': [
                        {'range': [0, 60], 'color': "#fee2e2"},
                        {'range': [60, 80], 'color': "#fef3c7"},
                        {'range': [80, 100], 'color': "#dcfce7"}
                    ]
                }
            ))

        efficiency_fig.update_layout(height=400)
        st.plotly_chart(efficiency_fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Main application function
def main():
    # This function is called when the script is run directly
    logger.info("AutoX Connect application started")

if __name__ == "__main__":
    main()
