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

class PredictionApp:
    def __init__(self):
        st.title("醫療預測系統")
        self.initialize_session_state()
        self.load_model()

    def initialize_session_state(self):
        if 'questionnaire_data' not in st.session_state:
            st.session_state.questionnaire_data = {}
        if 'image_data' not in st.session_state:
            st.session_state.image_data = None

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
                
                
                # 檢查必要欄位
                required_columns = [
                    "Education", "age", "gender", "BMI", "living_code", 
                    "exercise", "Hyperlipidemia",
                    "memory_mean(8)", "language_mean(9)", "space_mean(7)", 
                    "plan_ability_mean(5)", "organization_ability_mean(6)", 
                    "attention_mean(4)", "direction_mean(7)", 
                    "judgement_mean(5)", "care_mean(5)", 
                    "sympton_age", "sympton_speed"
                ]
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"缺少以下必要欄位: {missing_columns}")
                    return None
                
                # 取得資料的第一欄
                first_column = df.iloc[:, 0]  # 取得第一欄

                # 將第一欄加到 df_selected 中
                df_selected = df[required_columns]  # 只選取需要的欄位

                # 顯示資料
                st.write("資料的第一欄（通常是ID或號碼）:")
                st.write(first_column)
                st.write("資料 :")
                st.dataframe(df_selected)  # 顯示資料框

                # 儲存資料到 session_state 以便後續使用
                st.session_state.questionnaire_data = df_selected.iloc[0].to_dict()
                return df_selected
            
            except Exception as e:
                st.error(f"檔案讀取錯誤: {e}")
                return None
                
    def manual_input(self):
        required_columns = [
            "Education", "age", "gender", "BMI", "living_code", 
            "exercise", "Hyperlipidemia"
        ]
        questionnaire_columns = [
            "memory_mean(8)", "language_mean(9)", "space_mean(7)", 
            "plan_ability_mean(5)", "organization_ability_mean(6)", 
            "attention_mean(4)", "direction_mean(7)", 
            "judgement_mean(5)", "care_mean(5)", 
            "sympton_age", "sympton_speed"
        ]

        input_data = {}
        for col in required_columns + questionnaire_columns:
            input_data[col] = st.number_input(f"{col}:", value=0.0)
        
        st.session_state.questionnaire_data = input_data
        return input_data

    def upload_image(self):
        uploaded_file = st.file_uploader(
            "上傳醫療影像 (NIfTI 格式)", 
            type=['nii', 'nii.gz']
        )
        if uploaded_file is not None:
            try:
                # 將上傳文件保存到臨時路徑
                with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # 使用臨時路徑載入 NIfTI 影像
                nifti_img = nib.load(tmp_file_path)
                img_data = nifti_img.get_fdata()
                # 調整影像尺寸和形狀
                target_shape = (176, 224, 170)  # 注意添加通道維度
                img_data = np.resize(img_data, target_shape)
                # 刪除臨時文件
                os.unlink(tmp_file_path)
                
                st.session_state.image_data = img_data
                st.success("影像成功載入")
                return img_data
            except Exception as e:
                st.error(f"影像載入錯誤: {e}")
                return None

    def preprocess_data(self, img_data, questionnaire_data):
        # 標準化問卷資料
        # 將字典值轉換為浮點數
        X_table = np.array([
            [float(val) for val in questionnaire_data.values()]
        ])
        print(X_table)
        # 訓練集 X_table 的每個特徵的平均數和標準差
        mean = np.array([8.225, 70.7875, 0.3875, 24.05646937, 0.85, 0.525, 0.3125, 2.425875, 
                        1.72775, 1.795625, 2.01975, 1.957, 2.440625, 1.689, 1.695, 1.1775, 
                        66.925, 3.3])

        std = np.array([4.10175267, 7.06168137, 0.48717938, 3.9195689, 0.35707142, 0.49937461, 
                        0.46351241, 0.96991326, 0.73471078, 0.95668809, 0.83696024, 0.98179351, 
                        0.95952768, 0.70608356, 0.77328843, 0.48061809, 7.44441905, 1.3820275])

        X_table = (X_table - mean) / std
        # 載入之前儲存的最小值和最大值
        X_img_min = np.load('X_img_min.npy')
        X_img_max = np.load('X_img_max.npy')

        # 對新的資料進行標準化
        X_img_normalized = (img_data - X_img_min) / (X_img_max - X_img_min)
        return X_img_normalized, X_table

    def predict(self):
        if self.model is None:
            st.error("模型未載入，無法進行預測")
            return

        if st.session_state.image_data is None or not st.session_state.questionnaire_data:
            st.error("請先上傳影像和填寫問卷")
            return
        # 呼叫 preprocess_data 函數進行資料預處理
        X_img_normalized, X_table_normalized = self.preprocess_data(
            st.session_state.image_data, 
            st.session_state.questionnaire_data
        )
        # 擴展影像數據維度
        X_img_new = np.expand_dims(X_img_normalized, axis=0)
        X_table_new = np.array(X_table_normalized) # 這裡也應該擴展問卷資料維度
        
        # 調試輸出
        st.write("影像數據形狀:", X_img_new.shape)
        st.write("問卷數據形狀:", X_table_new.shape)
        #st.write("問卷數據:", X_table_new)
        try:
            y_pred_prob = self.model.predict([X_img_new, X_table_new])
            y_pred = (y_pred_prob > 0.5).astype(int)


            # 根據 y_pred 來顯示預測結果
            if y_pred[0][0] == 0:
                prediction_text = "預測結果顯示：Y=0 此為認知正常者 (MMSE > 26)"
            else:
                prediction_text = "預測結果顯示：Y=1 此為重度認知功能障礙者 (MMSE < 20)"

            st.write(prediction_text)
            st.write(f"預測 Y=1 的機率: {y_pred_prob[0][0]:.4f}")
        except Exception as e:
            st.error(f"預測過程發生錯誤: {e}")

    def run(self):
        st.sidebar.header("資料輸入")
        input_method = st.sidebar.radio(
            "選擇資料輸入方式", 
            ["上傳 CSV/XLSX","手動輸入"]
        )

        if input_method == "手動輸入":
            self.manual_input()
        else:
            self.upload_csv()

        self.upload_image()

        if st.button("開始預測"):
            self.predict()

def main():
    app = PredictionApp()
    app.run()

if __name__ == "__main__":
    main()
