import gradio as gr
import pickle
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Load ML model
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

# Store history
history = []

# ---------------- LOGIN FUNCTION ----------------
# ---------------- LOGIN FUNCTION ----------------
def login(user, pwd):
    if user == "lavanya" and pwd == "2318":
        return (
            "✅ Login Successful",
            gr.update(visible=False),
            gr.update(visible=True)
        )

    return (
        "❌ Wrong Username or Password",
        gr.update(visible=True),
        gr.update(visible=False)
    )


# ---------------- PREDICTION FUNCTION ----------------
def predict_message(msg):

    data = cv.transform([msg])

    result = model.predict(data)[0]

    prob = model.predict_proba(data)[0]

    spam_prob = prob[1] * 100
    ham_prob = prob[0] * 100

    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if result == "spam":
        final_result = "🚨 SPAM EMAIL"
        badge = "🔴 HIGH RISK"
        emoji = "🚨🚨🚨"

    else:
        final_result = "✅ SAFE EMAIL"
        badge = "🟢 SAFE"
        emoji = "😊✅"

    history.append({
        "Time": time,
        "Message": msg,
        "Result": final_result,
        "Spam %": round(spam_prob, 2),
        "Safe %": round(ham_prob, 2)
    })

    return (
        f"{emoji} {final_result}",
        badge,
        f"Spam: {spam_prob:.2f}% | Safe: {ham_prob:.2f}%"
    )


# ---------------- DOWNLOAD HISTORY ----------------
def download_history():

    df = pd.DataFrame(history)

    file_path = "history.csv"

    df.to_csv(file_path, index=False)

    return file_path


# ---------------- CHART FUNCTION ----------------
def chart():

    spam = sum(
        1 for x in history
        if "SPAM" in x["Result"]
    )

    ham = sum(
        1 for x in history
        if "SAFE" in x["Result"]
    )

    plt.figure(figsize=(5, 4))

    plt.bar(
        ["Spam", "Safe"],
        [spam, ham]
    )

    plt.title("Spam vs Safe Emails")

    plt.savefig("chart.png")

    return "chart.png"


# ---------------- UI ----------------
with gr.Blocks(theme=gr.themes.Soft()) as app:

    # LOGIN PAGE
    with gr.Column() as login_page:

        gr.Markdown("""
        # 🔐 Login Page
        ### Enter credentials to access AI Spam Detector
        """)

        user = gr.Textbox(
            label="Username"
        )

        pwd = gr.Textbox(
            label="Password",
            type="password"
        )

        login_btn = gr.Button(
            "🔓 Login"
        )

        login_msg = gr.Textbox(
            label="Status"
        )

    # MAIN APP
    with gr.Column(
        visible=False
    ) as main_app:

        # HOME TAB
        with gr.Tab("🏠 Home"):

            gr.Markdown("""
            # 📧 AI Email Spam Detector

            ## Detect Spam Emails Instantly

            🚀 AI Powered Detection

            🔒 Secure Email Analysis

            ⚡ Fast Prediction

            🎯 High Accuracy
            """)

        # DETECTION TAB
        with gr.Tab("📧 Detection"):

            msg = gr.Textbox(
                lines=6,
                placeholder="Enter email text here...",
                label="Input Message"
            )

            btn = gr.Button(
                "🔍 Check Email"
            )

            result = gr.Textbox(
                label="Prediction"
            )

            badge = gr.Textbox(
                label="Status Badge"
            )

            confidence = gr.Textbox(
                label="Confidence Score"
            )

            btn.click(
                predict_message,
                inputs=msg,
                outputs=[
                    result,
                    badge,
                    confidence
                ]
            )

        # ANALYTICS TAB
        with gr.Tab("📊 Analytics"):

            gr.Markdown(
                "## 📥 Download Prediction History"
            )

            download_btn = gr.Button(
                "⬇ Download CSV"
            )

            file_output = gr.File()

            download_btn.click(
                download_history,
                outputs=file_output
            )

            gr.Markdown(
                "## 📊 Spam vs Safe Graph"
            )

            chart_btn = gr.Button(
                "📈 Generate Chart"
            )

            chart_output = gr.Image()

            chart_btn.click(
                chart,
                outputs=chart_output
            )

        # ABOUT TAB
        with gr.Tab("ℹ️ About"):

            gr.Markdown("""
            # About Project

            This project uses Machine Learning
            and NLP techniques to classify
            emails as Spam or Safe.

            Features:
            - Spam Detection
            - Confidence Score
            - History Download
            - Analytics Dashboard
            """)

        # CONTACT TAB
        with gr.Tab("📞 Contact"):

            gr.Markdown("""
            # Contact

            Name: Lavanya

            Email: lavanyapobbathi23@gmail.com

            GitHub: https://github.com/Lavanyalavs23

            LinkedIn: https://www.linkedin.com/in/lavanya-pobbathi-0352a3315?utm_source=share_via&utm_content=profile&utm_medium=member_android
            """)

    # LOGIN BUTTON ACTION
    login_btn.click(
        login,
        inputs=[user, pwd],
        outputs=[
            login_msg,
            login_page,
            main_app
        ]
    )

# RUN APP
app.launch()
