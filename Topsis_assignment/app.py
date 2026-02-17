import streamlit as st
import pandas as pd
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

def send_email(sender_email, sender_password, receiver_email, subject, body, attachment_path):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    filename = os.path.basename(attachment_path)
    attachment = open(attachment_path, "rb")

    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        return True, "Email sent successfully!"
    except Exception as e:
        return False, str(e)

def topsis_logic(df, weights, impacts):
    # Matrix of alternatives (m) x criteria (n)
    # We assume first column is Name/ID
    matrix = df.iloc[:, 1:].values.astype(float)
    rows, cols = matrix.shape

    try:
        weight_list = [float(w) for w in weights.split(',')]
        impact_list = impacts.split(',')
    except ValueError:
        return None, "Weights must be numeric and separated by commas."
        
    if len(weight_list) != cols or len(impact_list) != cols:
        return None, "Number of weights, impacts and number of columns (from 2nd to last) must be the same."

    if not all(i in ['+', '-'] for i in impact_list):
        return None, "Impacts must be either +ve or -ve."

    # 1. Vector Normalization
    rss = np.sqrt(np.sum(matrix**2, axis=0))
    normalized_matrix = matrix / rss

    # 2. Weighted Normalization
    weighted_matrix = normalized_matrix * weight_list

    # 3. Ideal Best and Ideal Worst
    ideal_best = []
    ideal_worst = []

    for i in range(cols):
        if impact_list[i] == '+':
            ideal_best.append(np.max(weighted_matrix[:, i]))
            ideal_worst.append(np.min(weighted_matrix[:, i]))
        else:
            ideal_best.append(np.min(weighted_matrix[:, i]))
            ideal_worst.append(np.max(weighted_matrix[:, i]))
            
    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    # 4. Euclidean Distance
    dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))

    # 5. Performance Score
    score = dist_worst / (dist_best + dist_worst)

    # 6. Rank
    df['Topsis Score'] = score
    df['Rank'] = df['Topsis Score'].rank(ascending=False, method='min').astype(int)

    return df, None

st.set_page_config(page_title="Topsis Web Service", layout="centered")

st.title("Topsis Web Service - Aryan Thakur (102316004)")

st.sidebar.header("Email Configuration (Optional)")
sender_email = st.sidebar.text_input("Sender Email (Gmail)")
sender_password = st.sidebar.text_input("App Password", type="password")
st.sidebar.info("To send results via email, provide your Gmail address and App Password. Ensure 'Less secure access' is handled via App Passwords if 2FA is on.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Input Data Preview:")
        st.dataframe(df.head())
        
        if df.shape[1] < 3:
            st.error("Input file must contain three or more columns.")
        else:
            # Check numeric
            numeric_check = df.iloc[:, 1:].apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
            if not numeric_check.all():
                st.error("From 2nd to last columns must contain numeric values only.")
            else:
                weights = st.text_input("Weights (comma-separated, e.g., 1,1,1,1,1)")
                impacts = st.text_input("Impacts (comma-separated, e.g., +,+,+,+,+)")
                email_id = st.text_input("Email Id (Receiver)")
                
                if st.button("Submit"):
                    if not weights or not impacts or not email_id:
                        st.error("Please fill all fields.")
                    else:
                        result_df, error = topsis_logic(df.copy(), weights, impacts)
                        
                        if error:
                            st.error(error)
                        else:
                            st.success("Topsis calculation completed!")
                            st.write("Result:")
                            st.dataframe(result_df)
                            
                            # Save to a temporary file
                            result_file = "output_result.csv"
                            result_df.to_csv(result_file, index=False)
                            
                            # Send Email
                            if sender_email and sender_password:
                                success, msg = send_email(sender_email, sender_password, email_id, "TOPSIS Result", "Please find the attached TOPSIS result.", result_file)
                                if success:
                                    st.success(msg)
                                else:
                                    st.error(f"Failed to send email: {msg}")
                            else:
                                st.warning("Email credentials not provided. Email not sent.")
                                
                            # Download button
                            with open(result_file, "rb") as f:
                                st.download_button("Download Result CSV", f, file_name="result.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error reading file: {e}")

