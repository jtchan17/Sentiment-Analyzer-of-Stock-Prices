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

def check_user_n_password(username, password):
    query1 = text('SELECT * FROM dashboard.users WHERE BINARY username = :username AND password = :password')
    data = session.execute(query1, {"username": username, "password": make_hashes(password)}).fetchone()
    return data

def check_user(username):
    query1 = text('SELECT * FROM dashboard.users WHERE BINARY username = :username')
    data = session.execute(query1, {"username": username}).fetchone()
    return data

def update_userdata(username, password):
    query1 = text('UPDATE dashboard.users SET password = :password WHERE username = :username')
    session.execute(query1, {"username": username, "password": password})
    session.commit()

#######################################################################################################

if "role" not in st.session_state:
    st.session_state.role = None

if "username" not in st.session_state:
    st.session_state.username = None

ROLES = [None, "User", "Guest"]

#Sign up dialog to create new user account
@st.dialog("Create New Account")
def signup():

    new_user = st.text_input("Username", key='signupUsername')
    new_password = st.text_input("Password", key='signupPassword', type='password')

    #checking
    user_n_pw_check = check_user_n_password(new_user, new_password)
    user_check = check_user(new_user)

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

#A dialog for user who forgot the password
@st.dialog('Forgot your password?')
def forgotPassword():
    username = st.text_input('Please provide your username')
    existed_user = check_user(username)
    if existed_user:
        new_password = st.text_input('New Password', type='password')
        confirm_password = st.text_input('Confirm Password', type='password')
        confirm_button = st.button('Submit')
        if confirm_button:
            if new_password == confirm_password:
                update_userdata(username, make_hashes(confirm_password))
                st.success('Password update successfully. Please proceed to log in')
            else:
                 st.error('Password do not match')
    elif username == '':
         st.write('')
    else:
        st.warning('No such user exists.')

#Home function (show login page first)
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

        col1, col2 = st.columns([3.5, 1])
        with col1:
            login_Button = st.button("Login")
        with col2:
            forgotPassword_Button = st.button('Forgot Password')
            if forgotPassword_Button:
                 forgotPassword()
        st.write('')
        col3, col4, col5 = st.columns(3)
        with col4:
            signup_Button = st.button('New User? Click here.')
            if signup_Button:
                signup()

        if login_Button:
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

user_SA = st.Page(
     "users/sentiment_analyzer.py",
     title="Sentiment Analyzer",
     icon= ':material/analytics:',
)

user_analysis = st.Page(
    "users/analysis.py",
    title="Analysis",
    icon= ':material/analytics:',
)

guest = st.Page(
    "guest/guest_dashboard.py",
    title="Dashboard",
    icon=":material/home:",
    default=(role == "Guest"),
)

account_pages = [logout_page]
users_pages = [user, user_SA, settings_page]
guest_pages = [guest]

# st.title("Request manager")
# st.logo("images/horizontal_blue.png", icon_image="images/icon_blue.png")
page_dict = {}
if st.session_state.role in ["User"]:
    page_dict["User"] = users_pages
if st.session_state.role in ["Guest"]:
    page_dict["Guest"] = guest_pages

if len(page_dict) > 0:
    pg = st.navigation({"": account_pages} | page_dict)
else:
    pg = st.navigation([st.Page(home)])

pg.run()