import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import nibabel as nib
import tempfile
import os
import sys
import gdown
import re

class PredictionApp:
    def __init__(self):
        st.title("醫療預測系統")
        self.initialize_session_state()
        self.load_model()

    def initialize_session_state(self):
        session_vars = [
            'questionnaire_df', 'image_data', 'image_filenames', 
            'first_column', 'file_mapping', 'matched_data'
        ]
        for var in session_vars:
            if var not in st.session_state:
                st.session_state[var] = None

    def extract_number(self, x):
        """從字串結尾提取數字"""
        match = re.search(r'(\d+)$', str(x))
        return int(match.group(1)) if match else None
        
        
    def match_files(self, questionnaire_df, uploaded_files):
        """智能匹配問卷資料與影像檔案"""
        # 除錯：印出原始問卷資料
        st.write("問卷資料列數:", len(questionnaire_df))

        # 提取問卷資料的數字 ID
        questionnaire_df['ID'] = questionnaire_df.iloc[:, 0].apply(self.extract_number)
        
        # 提取影像檔案的數字 ID 和檔名
        image_ids = []
        image_names = []
        for file in uploaded_files:
            match = re.search(r'Img(\d+)', file.name)  # 假設檔名格式為 "Img123"
            id_value = int(match.group(1)) if match else None
            image_ids.append(id_value)
            image_names.append(file.name)

        # 除錯：印出影像檔案數量與影像 ID
        st.write("影像檔案數量:", len(uploaded_files))
        st.write("影像 ID:", image_ids)

        # 確保問卷資料與影像檔案數量一致
        if len(questionnaire_df) != len(uploaded_files):
            st.error("問卷資料與影像檔案數量不一致，請檢查資料！")
            return [], []

        # 確保 ID 與檔名完全匹配
        matched_data = []
        matched_images = []
        for idx, q_id in enumerate(questionnaire_df['ID']):
            # 找到與問卷 ID 完全匹配的影像索引
            try:
                image_match_idx = image_ids.index(q_id)
                matched_data.append(questionnaire_df.iloc[idx].to_dict())
                matched_images.append(uploaded_files[image_match_idx])
            except ValueError:
                # 若找不到對應的影像檔案
                st.error(f"問卷資料的 ID {q_id} 無法找到對應的影像檔案！")
                return [], []

        # 確認所有影像檔案都被匹配
        if len(matched_images) != len(uploaded_files):
            st.error("部分影像檔案未被匹配，請檢查資料！")
            return [], []

        return matched_data, matched_images

    def load_model(self):
        try:
            # Google Drive 檔案的分享連結
            drive_url = "https://drive.google.com/uc?id=1PeL_CoV3TldoyU9ZlPgtk4xH8V_QyomK"
            model_filename = "best_fold_model_f1_score.h5"
            model_path = os.path.join('data', model_filename)

            # 確保資料夾存在
            os.makedirs('data', exist_ok=True)

            # 如果模型檔案不存在，就下載
            if not os.path.exists(model_path):
                st.sidebar.info("正在下載模型檔案...")
                gdown.download(drive_url, model_path, quiet=False)

            # 載入模型
            self.model = load_model(model_path, compile=False)
            st.sidebar.success("模型已成功載入")
        except Exception as e:
            st.sidebar.error(f"模型載入失敗: {e}")
            self.model = None

    def upload_csv(self):
        uploaded_file = st.file_uploader("上傳 CSV/XLSX 檔案", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                # 取得第一欄（假設為ID）
                first_column = df.iloc[:, 0]
                st.session_state.first_column = first_column.tolist()

                # 檢查必要欄位
                required_columns = [
                    "Education", "age", "gender", "BMI", "living_code", 
                    "exercise", "Hyperlipidemia",
                    "memory_mean(8)", "language_mean(9)", "space_mean(7)", 
                    "plan_ability_mean(5)", "organization_ability_mean(6)", 
                    "attention_mean(4)", "direction_mean(7)", 
                    "judgement_mean(5)", "care_mean(5)", 
                    "symptom_age", "symptom_speed"
                ]
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"缺少以下必要欄位: {missing_columns}")
                    return None

                # 選取必要欄位的資料
                df_selected = df[required_columns]

                # 儲存資料
                st.session_state.original_questionnaire_df = df
                st.session_state.questionnaire_df = df_selected
                #st.session_state.questionnaire_data = df_selected.to_dict('records')
                
                st.write("資料的第一欄（ID）:")
                st.write(first_column)
                st.write("資料:")
                st.dataframe(df_selected)

                return df_selected

            except Exception as e:
                st.error(f"檔案讀取錯誤: {e}")
                return None


    def manual_input(self):
        # 定義欄位及其要求
        required_columns = [
            "Education", "age", "gender", "BMI", "living_code", 
            "exercise", "Hyperlipidemia"
        ]
        questionnaire_columns = [
            "memory_mean(8)", "language_mean(9)", "space_mean(7)", 
            "plan_ability_mean(5)", "organization_ability_mean(6)", 
            "attention_mean(4)", "direction_mean(7)", 
            "judgement_mean(5)", "care_mean(5)", 
            "symptom_age", "symptom_speed"
        ]

        # Session State 的鍵，避免覆蓋
        session_key = "manual_input_v1"

        # 初始化數據存儲區
        if session_key not in st.session_state:
            st.session_state[session_key] = {}

        # 指定需要整數格式的欄位
        integer_columns = ["symptom_age", "symptom_speed"]

        input_data = {}

        # 教育年數，整數格式
        input_data["Education"] = st.number_input(
            "Education (years, integer):", value=st.session_state[session_key].get("Education", 0), step=1, format="%d"
        )

        # 年齡，整數格式
        input_data["age"] = st.number_input(
            "Age (years old, integer):", value=st.session_state[session_key].get("age", 0), step=1, format="%d"
        )

        # 性別，只能選擇 0 或 1
        input_data["gender"] = st.selectbox(
            "Gender (0: female, 1: male):", options=[0, 1],
            index=st.session_state[session_key].get("gender", 0)
        )

        # 身高與體重，計算 BMI
        height = st.number_input(
            "Height (cm):", value=st.session_state[session_key].get("height", 0.0),  format="%.2f"
        )
        weight = st.number_input(
            "Weight (kg):", value=st.session_state[session_key].get("weight", 0.0),  format="%.2f"
        )
        if height > 0:
            input_data["BMI"] = round(weight / ((height / 100) ** 2), 2)
            st.write(f"BMI (calculated): {input_data['BMI']}")
        else:
            input_data["BMI"] = 0

        # 居住情況，只能選擇 0 或 1
        input_data["living_code"] = st.selectbox(
            "Living code (0: with others, 1: with spouse):", options=[0, 1],
            index=st.session_state[session_key].get("living_code", 0)
        )

        # 運動習慣，只能選擇 0 或 1
        input_data["exercise"] = st.selectbox(
            "Exercise (0: no regular exercise, 1: regular exercise):", options=[0, 1],
            index=st.session_state[session_key].get("exercise", 0)
        )

        # 高血脂，只能選擇 0 或 1
        input_data["Hyperlipidemia"] = st.selectbox(
            "Hyperlipidemia (0: no, 1: yes):", options=[0, 1],
            index=st.session_state[session_key].get("Hyperlipidemia", 0)
        )

        # 問卷欄位輸入
        for col in questionnaire_columns:
            if col in integer_columns:
                input_data[col] = st.number_input(
                    f"{col} (integer):", value=st.session_state[session_key].get(col, 0), step=1, format="%d"
                )  # 整數格式
            else:
                input_data[col] = st.number_input(
                    f"{col} (float):", value=st.session_state[session_key].get(col, 0.00), format="%.2f"
                )  # 小數點兩位格式

        # 在返回前顯示資料
        st.write("以下是輸入的資料：")
        df = pd.DataFrame([input_data])  # 將字典轉換為 DataFrame
        st.dataframe(df)  # 顯示資料框
        # 保存到 session state
        st.session_state.questionnaire_df = df

        return input_data
    
    def upload_image(self):
        if st.session_state.questionnaire_df is None:
            st.error("請先上傳問卷資料")
            return None
        # 檢查問卷資料是否存在
        if st.session_state.questionnaire_df.shape[0] == 1 :
            st.warning("問卷資料只能有一筆，將限制影像檔案只能上傳 1 個且不進行比對。")
            uploaded_file = st.file_uploader(
                "請上傳單一醫療影像檔案 (NIfTI 格式)", 
                type=['nii', 'nii.gz'],
                accept_multiple_files=False
            )

            if uploaded_file:
                try:
                    # 處理單一影像檔案
                    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    nifti_img = nib.load(tmp_file_path)
                    img_data = nifti_img.get_fdata()

                    target_shape = (176, 224, 170)
                    img_data = np.resize(img_data, target_shape)

                    os.unlink(tmp_file_path)

                    # 保存影像數據與檔名
                    st.session_state.image_data = [img_data]
                    st.session_state.image_filenames = [uploaded_file.name]

                    st.success(f"已成功上傳影像檔案：{uploaded_file.name}")
                    return [img_data]
                except Exception as e:
                    st.error(f"影像檔案 {uploaded_file.name} 載入錯誤: {e}")
                    return None

        
        else:
            # 問卷資料存在多個資料，進行多檔案上傳與比對
            uploaded_files = st.file_uploader(
                "上傳醫療影像 (NIfTI 格式)", 
                type=['nii', 'nii.gz'],
                accept_multiple_files=True
            )
            
            
            if uploaded_files:
                # 智能匹配
                matched_data, matched_images = self.match_files(
                    st.session_state.original_questionnaire_df, 
                    uploaded_files
                )

                # 檢查匹配結果
                if not matched_data:
                    st.error("無法找到匹配的檔案")
                    return None

                st.success(f"成功匹配 {len(matched_data)} 筆資料")

                # 儲存匹配結果
                st.session_state.matched_data = matched_data
                st.session_state.matched_images = matched_images

                # 處理影像
                image_data_list = []
                image_filename_list = []

                for uploaded_file in matched_images:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        nifti_img = nib.load(tmp_file_path)
                        img_data = nifti_img.get_fdata()
                        
                        target_shape = (176, 224, 170)
                        img_data = np.resize(img_data, target_shape)
                        
                        os.unlink(tmp_file_path)
                        
                        image_data_list.append(img_data)
                        image_filename_list.append(uploaded_file.name)
                    
                    except Exception as e:
                        st.error(f"影像 {uploaded_file.name} 載入錯誤: {e}")
                        return None
                
                st.session_state.image_data = image_data_list
                st.session_state.image_filenames = image_filename_list
                
                return image_data_list


    def preprocess_data(self, img_data, questionnaire_data, filename=None):
        # 標準化問卷資料
        X_table = questionnaire_data.values.astype(float).reshape(1, -1)

        # 訓練集 X_table 的每個特徵的平均數和標準差
        mean = np.array([8.225, 70.7875, 0.3875, 24.05646937, 0.85, 0.525, 0.3125, 2.425875, 
                        1.72775, 1.795625, 2.01975, 1.957, 2.440625, 1.689, 1.695, 1.1775, 
                        66.925, 3.3])

        std = np.array([4.10175267, 7.06168137, 0.48717938, 3.9195689, 0.35707142, 0.49937461, 
                        0.46351241, 0.96991326, 0.73471078, 0.95668809, 0.83696024, 0.98179351, 
                        0.95952768, 0.70608356, 0.77328843, 0.48061809, 7.44441905, 1.3820275])

        X_table = (X_table - mean) / std

        # 載入標準化參數
        x_img_max_path = os.path.join('X_img_max.npy')
        x_img_min_path = os.path.join('X_img_min.npy')

        X_img_max = np.load(x_img_max_path)
        X_img_min = np.load(x_img_min_path)

        # 對新的資料進行標準化
        X_img_normalized = (img_data - X_img_min) / (X_img_max - X_img_min)
        
        return X_img_normalized, X_table, filename
    
    def predict(self):
        if self.model is None:
            st.error("模型未載入，無法進行預測")
            return

        if st.session_state.image_data is None or st.session_state.questionnaire_df is None:
            st.error("請先上傳影像和填寫問卷")
            return

        # 準備結果列表
        results = []

        # 檢查是否為多筆資料
        if isinstance(st.session_state.image_data, list):
            # 多筆資料處理
            for img_data, (_, questionnaire_data), filename in zip(
                st.session_state.image_data, 
                st.session_state.questionnaire_df.iterrows(), 
                st.session_state.image_filenames
            ):
                
                # 資料預處理 - 直接傳入 Series
                X_img_normalized, X_table_normalized, file_id = self.preprocess_data(
                    img_data, questionnaire_data, filename
                )
                
                # 擴展維度
                X_img_new = np.expand_dims(X_img_normalized, axis=0)
                X_table_new = np.array(X_table_normalized)
                
                try:
                    # 模型預測
                    y_pred_prob = self.model.predict([X_img_new, X_table_new])
                    y_pred = (y_pred_prob > 0.5).astype(int)

                    # 準備結果
                    result = {
                        '檔名': file_id,
                        '預測結果': 'Y=0 為認知正常 (MMSE > 26)' if y_pred[0][0] == 0 else 'Y=1 為重度認知功能障礙 (MMSE < 20)',
                        'Y=1 機率': f"{y_pred_prob[0][0]:.4f}"
                    }
                    results.append(result)
                
                except Exception as e:
                    st.error(f"預測 {file_id} 時發生錯誤: {e}")

            # 顯示所有結果
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

        else:
            # 單筆資料處理
            questionnaire_data = st.session_state.questionnaire_df.iloc[0]
            X_img_normalized, X_table_normalized, _ = self.preprocess_data(
                st.session_state.image_data, 
                questionnaire_data
            )
            X_img_new = np.expand_dims(X_img_normalized, axis=0)
            X_table_new = np.array(X_table_normalized)
            
            try:
                y_pred_prob = self.model.predict([X_img_new, X_table_new])
                y_pred = (y_pred_prob > 0.5).astype(int)

                # 顯示單筆預測結果
                prediction_text = "預測結果顯示：Y=0 此為認知正常者 (MMSE > 26)" if y_pred[0][0] == 0 else "預測結果顯示：Y=1 此為重度認知功能障礙者 (MMSE < 20)"
                st.write(prediction_text)
                st.write(f"預測 Y=1 的機率: {y_pred_prob[0][0]:.4f}")
            
            except Exception as e:
                st.error(f"預測過程發生錯誤: {e}")

    def run(self):
        st.sidebar.header("資料輸入")
        input_method = st.sidebar.radio(
            "選擇資料輸入方式", 
            ["上傳 CSV/XLSX(可上傳多筆資料)","手動輸入"]
        )

        if input_method == "手動輸入":
            self.manual_input()
        else:
            column_descriptions = {
                "Education": "教育程度（在校幾年）",
                "age": "年齡",
                "gender": "性別 (0=女性, 1=男性)", 
                "BMI": "身體質量指標", 
                "living_code": "居住狀況（0=與他人同住，1=與配偶同住）", 
                "exercise": "是否常運動（0=否，1=是）", 
                "Hyperlipidemia": "是否有高血脂（0=否，1=是）", 
                "memory_mean(8)": "記憶能力測試平均分數", 
                "language_mean(9)": "語言能力測試平均分數", 
                "space_mean(7)": "空間感知能力測試平均分數", 
                "plan_ability_mean(5)": "規劃能力測試平均分數", 
                "organization_ability_mean(6)": "組織能力測試平均分數", 
                "attention_mean(4)": "注意力測試平均分數", 
                "direction_mean(7)": "方向感測試平均分數", 
                "judgement_mean(5)": "判斷力測試平均分數", 
                "care_mean(5)": "獨立生活能力測試平均分數", 
                "symptom_age": "失智症狀開始年齡", 
                "symptom_speed": "失智症狀發展速度"
            }
            # 顯示所有欄位及其說明（表格方式）
            st.write("需要的欄位及其說明(欄位順序不限)：")
            column_data = []
            for col, desc in column_descriptions.items():
                column_data.append([col, desc])

            # 使用表格顯示
            st.table(column_data)
            
            # 可以用Markdown來強調標題
            st.markdown("**提示：** 請根據欄位說明輸入資料。")

            self.upload_csv()

        self.upload_image()

        if st.button("開始預測"):
            self.predict()

def main():
    app = PredictionApp()
    app.run()

if __name__ == '__main__':
    main()
