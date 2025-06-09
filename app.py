import os
import tempfile
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
import pydub
from pydub import AudioSegment
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import streamlit as st

# =============================
# Streamlit Page Config & Title
# =============================
st.set_page_config(page_title="SixtyScan", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: #222; font-size: 72px;'>SixtyScan</h1>
    <p style='text-align: center; font-size: 20px; color: #444;'>ตรวจจับพาร์กินสันผ่านการวิเคราะห์เสียง</p>
""", unsafe_allow_html=True)

# =============================
# Custom CSS
# =============================
st.markdown("""
    <style>
        code, pre { display: none !important; }
        h1 { color: #222222; font-size: 64px; text-align: center; font-weight: bold; }
        p, .stMarkdown { color: #333333; font-size: 18px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# =============================
# Model Definition
# =============================
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

@st.cache_resource(show_spinner=False)
def load_model(path='best_resnet18.pth'):
    model = ResNet18Classifier()
    state = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

idx2label = {0: "NONPD", 1: "PD"}

# =============================
# Predict Function
# =============================
def predict(tensor):
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        return probs[0][1].item()

# =============================
# UI Mode Switch
# =============================
mode = st.radio("เลือกโหมด (Select mode):", ["Sample Mode", "Inference Mode"], index=1, horizontal=True)

# =============================
# Sample Mode
# =============================
if mode == "Sample Mode":
    st.sidebar.title("เลือกสเปกโตรแกรมตัวอย่าง")
    EXAMPLE_DIR = "exampleSpectrogram"
    examples = [f for f in sorted(os.listdir(EXAMPLE_DIR)) if f.endswith('.png')]
    selected_example = st.sidebar.selectbox("เลือกสเปกโตรแกรมตัวอย่าง", examples)
    if selected_example:
        image_path = os.path.join(EXAMPLE_DIR, selected_example)
        st.image(Image.open(image_path), caption=selected_example)

        # Predict sample button
        if st.button("🔍 Predict Sample"):
            img = Image.open(image_path).convert("RGB")
            tensor = preprocess(img).unsqueeze(0)
            prob = predict(tensor)
            percent = int(prob * 100)

            # Determine level and advice styling
            if percent <= 50:
                level = "ระดับต่ำ (Low)"
                label = "Non Parkinson"
                diagnosis = "ไม่เป็นพาร์กินสัน"
                box_color = "#e6f9e6"
                advice = """
                <div style='text-align: left;'>
                <p><b>คำแนะนำ:</b></p>
                <ul style='padding-left: 20px;'>
                    <li>ไม่มีอาการ: ควรตรวจปีละครั้ง</li>
                    <li>อาการเล็กน้อย (เช่น การเปลี่ยนแปลงเสียงเล็กน้อย อาการสั่นเล็กน้อย): ควรตรวจปีละ 2 ครั้ง</li>
                    <li>อาการเตือน (เช่น ส่งผลต่อการเคลื่อนไหว/การทำงานในแต่ละวัน): ควรตรวจ 2–4 ครั้งต่อปี</li>
                </ul>
                </div>
                """
            elif percent <= 75:
                level = "ปานกลาง (Moderate)"
                label = "Parkinson"
                diagnosis = "เป็นพาร์กินสัน"
                box_color = "#fff7e6"
                advice = """
                <div style='text-align: left;'>
                <p><b>คำแนะนำ:</b></p>
                <ul style='padding-left: 20px;'>
                    <li>ควรพบแพทย์เฉพาะทางระบบประสาท</li>
                    <li>ควรบันทึกอาการในแต่ละวัน</li>
                    <li>หากได้รับยา: ควรบันทึกผลข้างเคียงและประสิทธิภาพของยา</li>
                </ul>
                </div>
                """
            else:
                level = "สูง (High)"
                label = "Parkinson"
                diagnosis = "เป็นพาร์กินสัน"
                box_color = "#ffe6e6"
                advice = """
                <div style='text-align: left;'>
                <p><b>คำแนะนำ:</b></p>
                <ul style='padding-left: 20px;'>
                    <li>ควรพบแพทย์เฉพาะทางระบบประสาทโดยเร็วที่สุด</li>
                    <li>ควรบันทึกอาการในแต่ละวัน</li>
                    <li>หากได้รับยา: ควรบันทึกผลข้างเคียงและผลลัพธ์ของยา</li>
                </ul>
                </div>
                """

            st.markdown(f"""
                <div style='background-color:{box_color}; padding: 20px; border-radius: 10px; font-size: 18px; color: #000000;'>
                    <div style='text-align: center; font-size: 26px; font-weight: bold; margin-bottom: 12px;'>{label}:</div>
                    <h3 style='text-align: left;'>ระดับความน่าจะเป็น: {level}</h3>
                    <h3 style='text-align: left;'>ความน่าจะเป็นของพาร์กินสัน: {percent}%</h3>
                    <div style='height: 28px; background: linear-gradient(to right, green, yellow, red); border-radius: 6px; margin-bottom: 12px; position: relative;'>
                        <div style='position: absolute; left: {percent}%; top: 0; bottom: 0; width: 3px; background-color: black;'></div>
                    </div>
                    <h3 style='text-align: left;'>ผลการวิเคราะห์: {diagnosis}</h3>
                    {advice}
                </div>
            """, unsafe_allow_html=True)

# =============================
# Inference Mode
# =============================
if mode == "Inference Mode":
    st.markdown("""
        <ul style='font-size:18px; text-align:left;'>
            <li>กรุณาอัพโหลดการออกเสียงตามคำสั่ง</li>
            <li>พยางค์: ออกเสียงอย่างชัดเจนพยางค์และ 5-8 วินาที โดยสามารถอัพโหลดเป็นไฟล์แยก</li>
        </ul>
        <p style='font-size:18px; text-align:left; margin-left: 24px;'> "ไอ"  "อำ"  "เอ"  "อือ"  "อี"  "อำ"  "อา"  "ยู"</p>
        <ul style='font-size:18px; text-align:left;'>
            <li>พาทาคา: กรุณาออกเสียงแต่ละพยางค์ 2 วินาทีโดยรวมทั้งหมด 6 วินาที</li>
        </ul>
        <p style='font-size:18px; text-align:left; margin-left: 24px;'> "พา-ทา-คา"</p>
        <ul style='font-size:18px; text-align:left;'>
            <li>ประโยค: กรุณาอ่านประโยคอย่างชัดเจน</li>
        </ul>
        <p style='font-size:18px; text-align:left; margin-left: 24px;'> "วันนี้อากาศแจ่มใสนกร้องเสียงดังฟังชัด"</p>
        <ul style='font-size:18px; text-align:left;'>
            <li>เมื่ออัปโหลดครบแล้ว ให้กดปุ่ม Predict</li>
        </ul>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader("อัปโหลดไฟล์เสียง (wav/mp3/aac/m4a)", type=["wav", "mp3", "aac", "m4a"], accept_multiple_files=True)

    if st.button("🔍 Predict"):
        if not uploaded_files or len(uploaded_files) == 0:
            st.warning("กรุณาอัปโหลดไฟล์เสียงก่อน")
        elif len(uploaded_files) > 10:
            st.warning("กรุณาอัปโหลดไม่เกิน 10 ไฟล์")
        else:
            all_probs = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    audio = AudioSegment.from_file(file)
                    audio.export(tmp.name, format="wav")
                    y, sr = librosa.load(tmp.name, sr=16000)
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    mel_spec_db = librosa.util.fix_length(mel_spec_db, size=128, axis=1)
                    mel_spec_resized = Image.fromarray(mel_spec_db).resize((224, 224))
                    image = np.array(mel_spec_resized)
                    image = np.stack([image, image, image], axis=0)
                    image = (image - image.min()) / (image.max() - image.min())
                    tensor = torch.tensor(image).float().unsqueeze(0)
                    prob = predict(tensor)
                    all_probs.append(prob)

            final_prob = np.mean(all_probs)
            percent = int(final_prob * 100)

            if percent <= 50:
                level = "ระดับต่ำ (Low)"
                label = "Non Parkinson"
                diagnosis = "ไม่เป็นพาร์กินสัน"
                box_color = "#e6f9e6"
                advice = """
                <div style='text-align: left;'>
                <p><b>คำแนะนำ:</b></p>
                <ul style='padding-left: 20px;'>
                    <li>ไม่มีอาการ: ควรตรวจปีละครั้ง</li>
                    <li>อาการเล็กน้อย (เช่น การเปลี่ยนแปลงเสียงเล็กน้อย อาการสั่นเล็กน้อย): ควรตรวจปีละ 2 ครั้ง</li>
                    <li>อาการเตือน (เช่น ส่งผลต่อการเคลื่อนไหว/การทำงานในแต่ละวัน): ควรตรวจ 2–4 ครั้งต่อปี</li>
                </ul>
                </div>
                """
            elif percent <= 75:
                level = "ปานกลาง (Moderate)"
                label = "Parkinson"
                diagnosis = "เป็นพาร์กินสัน"
                box_color = "#fff7e6"
                advice = """
                <div style='text-align: left;'>
                <p><b>คำแนะนำ:</b></p>
                <ul style='padding-left: 20px;'>
                    <li>ควรพบแพทย์เฉพาะทางระบบประสาท</li>
                    <li>ควรบันทึกอาการในแต่ละวัน</li>
                    <li>หากได้รับยา: ควรบันทึกผลข้างเคียงและประสิทธิภาพของยา</li>
                </ul>
                </div>
                """
            else:
                level = "สูง (High)"
                label = "Parkinson"
                diagnosis = "เป็นพาร์กินสัน"
                box_color = "#ffe6e6"
                advice = """
                <div style='text-align: left;'>
                <p><b>คำแนะนำ:</b></p>
                <ul style='padding-left: 20px;'>
                    <li>ควรพบแพทย์เฉพาะทางระบบประสาทโดยเร็วที่สุด</li>
                    <li>ควรบันทึกอาการในแต่ละวัน</li>
                    <li>หากได้รับยา: ควรบันทึกผลข้างเคียงและผลลัพธ์ของยา</li>
                </ul>
                </div>
                """

            st.markdown(f"""
                <div style='background-color:{box_color}; padding: 20px; border-radius: 10px; font-size: 18px; color: #000000;'>
                    <div style='text-align: center; font-size: 26px; font-weight: bold; margin-bottom: 12px;'>{label}:</div>
                    <h3 style='text-align: left;'>ระดับความน่าจะเป็น: {level}</h3>
                    <h3 style='text-align: left;'>ความน่าจะเป็นของพาร์กินสัน: {percent}%</h3>
                    <div style='height: 28px; background: linear-gradient(to right, green, yellow, red); border-radius: 6px; margin-bottom: 12px; position: relative;'>
                        <div style='position: absolute; left: {percent}%; top: 0; bottom: 0; width: 3px; background-color: black;'></div>
                    </div>
                    <h3 style='text-align: left;'>ผลการวิเคราะห์: {diagnosis}</h3>
                    {advice}
                </div>
            """, unsafe_allow_html=True)

# =============================
# Clear Button
# =============================
if st.button("Clear"):
    st.session_state.clear()
    st.experimental_rerun()
