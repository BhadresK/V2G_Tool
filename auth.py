import hmac
import streamlit as st

def require_login() -> str:
    if st.session_state.get("auth_user"):
        return st.session_state["auth_user"]

    users = st.secrets.get("users", {})

    st.title("S.KOe COOL 2.0 - V2G Optimisation")
    st.caption("Nur zur internen Verwendung | For internal use only")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Sign in"):
        expected = users.get(username.strip(), "")
        if expected and hmac.compare_digest(str(expected), password):
            st.session_state["auth_user"] = username.strip()
            st.rerun()
        else:
            st.error("Invalid credentials.")

    st.stop()