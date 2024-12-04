import streamlit as st
from sqlalchemy.sql import text
import hashlib

conn = st.connection('mysql', "sql")
session = conn.session

def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def edit_userpassword(username,password):
	query = text('UPDATE dashboard.users SET password = :password WHERE username = :username')
	session.execute(query, {"username": username, "password": password})
	session.commit()
     
st.header("Update Your Profile")
st.write(f"You are logged in as {st.session_state.role}.")

# Form for profile updates
with st.form(key='profile_form'):
    # User's current name and email (pre-filled)
    # name = st.text_input("Name", value="John Doe")
    username = st.text_input("Username", value=f"{st.session_state.username}", disabled=True)
    # email = st.text_input("Email", value="john.doe@example.com")

    # Password change fields
    new_password = st.text_input("New Password", type='password')
    confirm_password = st.text_input("Confirm New Password", type='password')

    # Profile picture upload
    # profile_pic = st.file_uploader("Upload Profile Picture", type=["png", "jpg", "jpeg"])

    # Bio text area
    # bio = st.text_area("Bio", "This is my bio...")

    # Submit button 
    submit_button = st.form_submit_button("Save Changes")

    if submit_button:
        # Simple validation for demonstration
        if new_password and new_password != confirm_password:
            st.error("Passwords do not match.")
        else:
            edit_userpassword(f'{st.session_state.username}', make_hashes(confirm_password))
            st.success("Profile updated successfully!")
            # Add logic to save changes to the database or backend system