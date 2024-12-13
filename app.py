# Import necessary libraries
import os
import pandas as pd
from pandasai.connectors import PandasConnector
from pandasai import SmartDataframe
from pandasai.llm import BambooLLM, OpenAI
from langchain_groq.chat_models import ChatGroq 
import chainlit as cl

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Define a dictionary to store field descriptions
global field_descriptions
field_descriptions = {
'sales_order_number':'Unique identifier for each sales order.',
'sales_order_date':'The date and time when the sales order was placed. (e.g., Friday, August 25, 2017)',
'sales_order_date_day_of_week':'The day of the week when the sales order was placed (e.g., Monday, Tuesday).',
'sales_order_date_month':'The month when the sales order was placed (e.g., January, February).',
'sales_order_date_day':'The day of the month when the sales order was placed (1-31).',
'sales_order_date_year':'The year when the sales order was placed (e.g., 2022).',
'quantity':'The number of units sold in the sales order.',
'unit_price':'The price per unit of the product sold.',
'total_sales':'The total sales amount for the sales order (quantity * unit price).',
'cost':'The total cost associated with the products sold in the sales order.',
'product_key':'Unique identifier for the product sold.',
'product_name':'The name of the product sold.',
'reseller_key':'Unique identifier for the reseller.',
'reseller_name':'The name of the reseller.',
'reseller_business_type':'The type of business of the reseller (e.g., Warehouse, Value Reseller, Specialty Bike Shop).',
'reseller_city':'The city where the reseller is located.',
'reseller_state':'The state where the reseller is located.',
'reseller_country':'The country where the reseller is located.',
'employee_key':'Unique identifier for the employee associated with the sales order.',
'employee_id':'The ID of the employee who processed the sales order.',
'salesperson_fullname':'The full name of the salesperson associated with the sales order.',
'salesperson_title':'The title of the salesperson (e.g., North American Sales Manager, Sales Representative).',
'email_address':'The email address of the salesperson.',
'sales_territory_key':'Unique identifier for the sales territory for the actual sale. (e.g. 3)',
'assigned_sales_territory':'List of sales_territory_key separated by comma assigned to the salesperson. (e.g., 3,4)',
'sales_territory_region':'The region of the sales territory. US territory broken down in regions. International regions listed as country name (e.g., Northeast, France).',
'sales_territory_country':'The country associated with the sales territory.',
'sales_territory_group':'The group classification of the sales territory. (e.g., Europe, North America, Pacific)',
'target':'The sales target set for the salesperson or territory for the particular month when sales_order_date was placed.',
'target_date':'The date by which the sales target should be achieved. All dates are 1st day of the month. (e.g., Friday, August 1, 2017)',
'target_date_day_of_week':'The day of the week for the target date.',
'target_date_month':'The month for the target date (e.g., January, February).',
'target_date_day':'The day of the month for the target date. All dates are 1st day of the month. Value is set to 1.',
'target_date_year':'The year for the target date (e.g., 2022).',
}

# Define a function to handle the start of a chat
@cl.on_chat_start
async def start():
    # Initialize the user session with a message history
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )

# Define a function to handle incoming messages
@cl.on_message
async def main(message: cl.Message):
    # Retrieve the message history from the user session
    message_history = cl.user_session.get("message_history")
    
    # Add the incoming message to the message history
    message_history.append({"role": "user", "content": message.content})

    # Load the sales data from an Excel file
    df = pd.read_excel("adventureworks_2022_denormalized.xlsx")

    # Create a Pandas connector with the loaded data and field descriptions
    connector = PandasConnector({"original_df": df}, field_descriptions=field_descriptions)

    # Define the models to use for the chat
    llama = "llama3-70b-8192"
    mistral = "mixtral-8x7b-32768"
    oai = "OpenAI"
    model = llama

    # Create a language model based on the selected model
    if model in ("llama3-70b-8192", "mixtral-8x7b-32768"):
        llm = ChatGroq(model=model, api_key=os.environ["GROQ_API_KEY"])
    elif model == "OpenAI":
        llm = OpenAI()
    else:
        model = "BambooLLM"
        llm = BambooLLM()

    # Send an intro message with the selected model
    intro_message = "Model = " + model
    await cl.Message(content=intro_message).send()

    # Create a SmartDataframe with the connector and language model
    df = SmartDataframe(
        connector, 
        config={
            "verbose": True,
            "enable_cache": False,
            "custom_whitelisted_dependencies": ["collections"],
            "llm": llm
        }
    )

    # Get the question from the incoming message
    question = message.content

    # Get the response from the SmartDataframe
    response = df.chat(question)

    # Send the response as a message
    msg = cl.Message(content=response)
    await msg.send()

    # Update the message history with the response
    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()