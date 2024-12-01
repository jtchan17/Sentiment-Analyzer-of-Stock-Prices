import streamlit as st
import pandas as pd
from sqlalchemy.sql import text
import os

# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer Dashboard",
    page_icon="📈",
    layout="wide")

#######################################################################################################
# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

# DB Management
conn = st.connection('mysql', "sql")
session = conn.session

#######################################################################################################
# DB  Functions
def create_usertable():
	session.execute('CREATE TABLE IF NOT EXISTS dashboard.users (username TEXT,password TEXT)')

def add_userdata(username,password):
	query = text('INSERT INTO dashboard.users (username,password) VALUES (:username, :password)')
	session.execute(query, {"username": username, "password": password})
	session.commit()

def login_user(username,password):
    query = text('SELECT * FROM dashboard.users WHERE username = :username AND password = :password')
    data = session.execute(query, {"username": username, "password": password}).fetchone()
    return data if data else None

def view_all_users():
	data = session.execute(text('SELECT * FROM dashboard.users'))
	return data

def clear_fields(username, password):
    if username not in st.session_state:
        st.session_state[username] = ""

    if password not in st.session_state:
        st.session_state[password] = ""
#######################################################################################################

if "role" not in st.session_state:
    st.session_state.role = None

if "username" not in st.session_state:
    st.session_state.username = None

ROLES = [None, "User", "Guest", "Admin"]

# def login():
#     # st.title('Welcome to Sentiment Analyzer Dashboard')
#     st.subheader("Log in", divider=True)
#     # role = st.selectbox("Choose your role", ROLES)
#     # loginUsername = ''
#     # if role == "User":
#         # st.session_state["current_tab"] = "Login"
#         # signup()
#     keyLoginUsername  = 'loginUsername'
#     keyLoginPassword = 'loginPassword'
#     loginUsername = st.text_input("Username", key=keyLoginUsername)
#     loginPassword = st.text_input("Password", key=keyLoginPassword, type='password')
    
#         # Clear fields when switching to this tab
#     if st.session_state.get("current_tab") != "Login":
#         clear_fields(keyLoginUsername, keyLoginPassword)

#     col1, col2, col3 = st.columns([3.5, 5, 5])
#     with col1:
#         login_Button = st.button("Login")
#     with col2:
#         forgotPassword_Button = st.button('Forgot Password')
#     with col3:
#          signup_Button = st.button('New User? Click here.')
#          if signup_Button:
#               signup()

#     if login_Button:
#         # if password == '12345':
#         # create_usertable()
#         hashed_pswd = make_hashes(loginPassword)
#         result = login_user(loginUsername,check_hashes(loginPassword,hashed_pswd))
#         print("Login result:", result)
#         if result is None or not result:
#             st.warning("Incorrect Username/Password") 
#         else:
#             st.success("Logged In as {}".format(loginUsername))
#             st.session_state.role = role
#             st.session_state.username = loginUsername
#             st.rerun()  
            

        # with signUp:
        #     st.subheader("Create New Account")
        #     new_user = st.text_input("Username", key='signupUsername')
        #     new_password = st.text_input("Password", key='signupPassword', type='password')
        #     query = text('SELECT * FROM dashboard.users WHERE username = :username AND password = :password')
        #     result = session.execute(query, {"username": new_user, "password": make_hashes(new_password)})
        #     signupButton = st.button("Sign up")
        #     #check if the account has been created before
        #     if signupButton:
        #         if result:
        #           add_userdata(new_user,make_hashes(new_password))
        #           st.success("You have successfully created a valid Account")
        #           st.info("Go to Login Menu to login")
        #         else:
        #             st.warning('This account has been created.')
    # else:     
    #     if st.button("Log in"):
    #         st.session_state.role = role
    #         st.session_state.username = loginUsername
    #         st.rerun()

@st.dialog("Create New Account")
def signup():

    # if 'signupUsername' not in st.session_state:
    #     st.session_state.signupUsername = ""
    # if 'signupPassword' not in st.session_state:
    #     st.session_state.signupPassword = ""

    new_user = st.text_input("Username", key='signupUsername')
    new_password = st.text_input("Password", key='signupPassword', type='password')

    #checking
    query1 = text('SELECT * FROM dashboard.users WHERE username = :username AND password = :password')
    user_n_pw_check = session.execute(query1, {"username": new_user, "password": make_hashes(new_password)}).fetchone()
    query2 = text('SELECT * FROM dashboard.users WHERE username = :username')
    user_check = session.execute(query2, {"username": new_user}).fetchone()

    signupButton = st.button("Sign up")

    if signupButton:
        if new_user != '' and new_password != '':
            if user_n_pw_check: #account already existed
                st.warning('This account has been created.')
            elif user_check: #username already exixted
                st.warning('Username has been used. Please fill in a new username.')
            else:              
                add_userdata(new_user,make_hashes(new_password))
                st.success("You have successfully created a valid Account")
                st.info("Go to Login Menu to login")
                # Clear the session state after successful signup
            # st.session_state.signupUsername = ""
            # st.session_state.signupPassword = ""
        else:
            st.warning('Please fill in the empty fields.')

    # Check if the dialog is closed and reset the input values
    # if not st.session_state.get('dialog_open', True):
    #     st.session_state.signupUsername = ""
    #     st.session_state.signupPassword = ""

def home():
    st.title('Welcome to Sentiment Analyzer Dashboard')
    st.divider()
    role = st.selectbox("Choose your role", ROLES)
    # st.header("", divider="blue")
    loginUsername = ''
    if role == "User":
        st.subheader("Log in", divider=True)
        keyLoginUsername  = 'loginUsername'
        keyLoginPassword = 'loginPassword'
        loginUsername = st.text_input("Username", key=keyLoginUsername)
        loginPassword = st.text_input("Password", key=keyLoginPassword, type='password')

        col1, col2, col3 = st.columns([3.5, 5, 5])
        with col1:
            login_Button = st.button("Login")
        with col2:
            forgotPassword_Button = st.button('Forgot Password')
        with col3:
            signup_Button = st.button('New User? Click here.')
            if signup_Button:
                signup()

        if login_Button:
            # if password == '12345':
            # create_usertable()
            hashed_pswd = make_hashes(loginPassword)
            result = login_user(loginUsername,check_hashes(loginPassword,hashed_pswd))
            print("Login result:", result)
            if result is None or not result:
                st.warning("Incorrect Username/Password") 
            else:
                st.success("Logged In as {}".format(loginUsername))
                st.session_state.role = role
                st.session_state.username = loginUsername
                st.rerun()  
    else:
        if st.button("Log in"):
            st.session_state.role = role
            st.session_state.username = loginUsername
            st.rerun()

def logout():
    st.session_state.role = None
    st.session_state.username = None
    st.rerun()

role = st.session_state.role
username = st.session_state.username

logout_page = st.Page(logout, title="Log out", icon=":material/logout:")
settings_page = st.Page("users/edit_profile.py", title="Edit Profile", icon=":material/edit:")

user= st.Page(
    "users/user_dashboard.py",
    title="Dashboard",
    icon=":material/home:",
    default=(role == "User"),
)
# request_2 = st.Page(
#     "request/request_2.py", title="Request 2", icon=":material/bug_report:"
# )

user_SA = st.Page(
     "users/sentiment_analyzer.py",
     title="Sentiment Analyzer",
     icon= ':material/analytics:',
)

guest = st.Page(
    "guest/guest_dashboard.py",
    title="Dashboard",
    icon=":material/home:",
    default=(role == "Guest"),
)
# respond_2 = st.Page(
#     "respond/respond_2.py", title="Respond 2", icon=":material/handyman:"
# )
admin_1 = st.Page(
    "admin/admin_dashboard.py",
    title="Dashboard",
    icon=":material/person:",
    default=(role == "Admin"),
)
# admin_2 = st.Page("admin/admin_2.py", title="Admin 2", icon=":material/security:")

account_pages = [logout_page]
users_pages = [user, user_SA, settings_page]
guest_pages = [guest]
admin_pages = [admin_1]

# st.title("Request manager")
# st.logo("images/horizontal_blue.png", icon_image="images/icon_blue.png")
page_dict = {}
if st.session_state.role in ["User", "Admin"]:
    page_dict["User"] = users_pages
if st.session_state.role in ["Guest", "Admin"]:
    page_dict["Guest"] = guest_pages
if st.session_state.role == "Admin":
    page_dict["Admin"] = admin_pages

if len(page_dict) > 0:
    pg = st.navigation({"": account_pages} | page_dict)
else:
    pg = st.navigation([st.Page(home)])

pg.run()