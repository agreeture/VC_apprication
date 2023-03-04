from tkinter import *
from tkinter import ttk

import wave
import sys
import time
import threading

import sounddevice as sd
import numpy as np

import inference


class client(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        
        # 録音設定
        self.nchannels=1
        self.fs=24000
        self.blocksize=1024
        
        # デバイス設定
        self.default_mic_index = 0
        self.default_speaker_index = 0
        self.mic_list, self.mic_dict, self.speaker_list, self.speaker_dict = self.get_device_index()
        
        # Streamの設定
        self.stream = sd.Stream(
            samplerate=self.fs,
            channels=self.nchannels,
            blocksize=self.blocksize,     # The number of frames passed to the stream callback function.
            dtype='float32'
        )
        
        self.wav = None
        
        # flag設定
        self.convert_flag = None
        self.vc_end_flag = None
        self.application_end_flag = None
        self.prev_data_flag = False
        
        # max list length
        self.max_list_len = 10
        self.overlapped_length = 100
        self.prev_data = None
        
        # set up list
        self.record_list = []
        self.converted_list = []
        for i in range(3):
            self.converted_list.append(np.zeros((self.blocksize, 1)))
        
        # parameter for VC
        speaker_A_id = "me"
        speaker_B_id = "jvs063"
        preprocessed_data_dir = r"preprocessed_data"
        ckpt_dir = r"maskcyclegan_result"
        model_name = "generator_A2B"
        load_epoch = 5300
        
        # VCモデル読み込み 
        self.VC_model = inference.VC(self.fs, speaker_A_id, speaker_B_id, 
                                    preprocessed_data_dir, ckpt_dir, model_name, load_epoch)
        
        # ウィンドウの設定
        self.master.title("VC")
 
        # 実行内容
        #self.pack()
        self.create_widget()
    
    # create_widgetメソッドを定義
    def create_widget(self):
        # デバイス選択frame作成
        self.frame_device = ttk.Frame(
            self.master,
            padding=(5),
            relief='ridge')
        self.frame_device.grid()
    
        # 録音デバイス選択プルダウンラベル
        self.mic_string = StringVar() 
        self.label_mic = ttk.Label(
            self.frame_device,
            text=u"マイク選択",
            font=(10)
            )
        self.label_mic.grid(row=0, column=0)
        
        # 録音デバイス選択プルダウン本体
        self.cb_mic = ttk.Combobox(
            self.frame_device, 
            textvariable=self.mic_string, 
            values=self.mic_list, 
            width=30,
            font=(10)
            )
        self.cb_mic.set(self.mic_list[self.default_mic_index])
        self.cb_mic.grid(row=0, column=1)
    
        # 出力デバイス選択プルダウンラベル
        self.speaker_string = StringVar() 
        self.label_speaker = ttk.Label(
            self.frame_device,
            text=u"出力デバイス選択",
            font=(10)
            )
        self.label_speaker.grid(row=1, column=0)
        
        # 出力デバイス選択プルダウン本体
        self.cb_speaker = ttk.Combobox(
            self.frame_device, 
            textvariable=self.speaker_string, 
            values=self.speaker_list, 
            width=30,
            font=(10)
            )
        self.cb_speaker.set(self.speaker_list[self.default_speaker_index])
        self.cb_speaker.grid(row=1, column=1)
    
        # 録音frame作成
        self.frame_record = ttk.Frame(
            self.master,
            padding=(5),
            relief='ridge')
        self.frame_record.grid()
        
        # 録音ボタン
        self.button_start_record = ttk.Button(
            self.frame_record,
            text=u'開始',
            compound=TOP,
            padding=(4),
            command=self.clicked_start_recording)
        self.button_start_record.grid(row=0, column=2, rowspan=2)
        
        # 録音停止ボタン
        self.button_stop_record = ttk.Button(
            self.frame_record,
            text=u'終了',
            compound=TOP,
            padding=(4),
            command=self.clicked_stop_recording)
        self.button_stop_record.grid(row=0, column=3, rowspan=2)

    def run_application(self):
        if self.convert_flag == True:
            # 声質変換開始
            if self.stream.read_available > self.blocksize:
                indata = self.input_stream()
                print("prev_data_flag : True")
                indata = indata.reshape(self.blocksize)
            else :
                print("prev_data_flag : False")
                indata = np.zeros((self.blocksize))
            print("indata shape : ",end="")
            print(indata.shape)
            converted_wav = self.convert_wav(indata)
            if self.prev_data_flag == False:
                self.prev_data = np.zeros((self.blocksize))
                marged_wav = self.marge_wav(converted_wav, self.prev_data, 0)
            else :
                marged_wav = self.marge_wav(converted_wav, self.prev_data, self.overlapped_length)
            
            self.wav = marged_wav
            thread_output_wav = threading.Thread(target=self.output_stream)
            thread_output_wav.start()
            
            #self.output_stream(marged_wav)
            self.prev_data = converted_wav
            self.prev_data_flag = True
            
        if self.vc_end_flag == True:
            # VC終了
            self.prev_data_flag = False
            
        if self.application_end_flag == True:
            # アプリケーション終了
            print("finish")
                

    def clicked_start_recording(self):
        self.button_start_record.state(['pressed'])
        self.start_stream()
        self.convert_flag = True
        self.vc_end_flag = False
        
    def clicked_stop_recording(self):
        self.button_stop_record.state(['pressed'])
        self.button_start_record.state(['!pressed'])
        self.stream.stop()
        self.convert_flag = False
        self.vc_end_flag = True
        self.button_stop_record.state(['!pressed'])
    
    def input_stream(self):
        input_data = self.stream.read(self.blocksize)
        return input_data[0]
    
    #def output_stream(self, wav):
        #wav = wav.astype(np.float32)
    def output_stream(self):
        self.stream.write(self.wav.astype(np.float32))
    
    def start_stream(self):
        self.stream.start()
    
    def get_device_index(self):
        ''' マイクチャンネルのindexをリストで取得する '''
        # 最大入力チャンネル数が0でない項目をマイクチャンネルとしてリストに追加
        device_list = sd.query_devices()
        mic_list = []
        mic_number_list = []
        speaker_list = []
        speaker_number_list = []
        mic_number = 0
        speaker_number = 0
        for i in range(len(device_list)):
            num_of_input_ch = device_list[i]['max_input_channels']
            num_of_output_ch = device_list[i]['max_output_channels']
            hostapi = device_list[i]['hostapi']
            
            if num_of_input_ch != 0 and hostapi == 0:
                mic_list.append(device_list[i]['name'])
                mic_number_list.append(device_list[i]['index'])
                if i == sd.default.device[0]:
                    self.default_mic_index = mic_number
                mic_number = mic_number + 1
            
            elif num_of_output_ch != 0 and hostapi == 0:
                speaker_list.append(device_list[i]['name'])
                speaker_number_list.append(device_list[i]['index'])
                if i == sd.default.device[1]:
                    self.default_speaker_index = speaker_number
                speaker_number = speaker_number + 1
        
        mic_dict = dict(zip(mic_list, mic_number_list))
        speaker_dict = dict(zip(speaker_list, speaker_number_list))
        return mic_list, mic_dict, speaker_list, speaker_dict
        
        
    def convert_wav(self, wav):
        target_wav = self.VC_model.inference(wav)
        return target_wav.to('cpu').detach().numpy().copy()[:self.blocksize].reshape(self.blocksize, 1)
        
    def marge_wav(self, now_wav, prev_wav, overlap_length):
        """
        生成したwavデータを前回生成したwavデータとoverlap_lengthだけ重ねてグラデーション的にマージします
        終端のoverlap_lengthぶんは次回マージしてから再生するので削除します
        Parameters
        ----------
        now_wav: 今回生成した音声wavデータ
        prev_wav: 前回生成した音声wavデータ
        overlap_length: 重ねる長さ
        """
        if overlap_length == 0:
            return now_wav
        gradation = np.arange(overlap_length) / overlap_length
        now_head = now_wav[:overlap_length]
        prev_tail = prev_wav[-overlap_length:]
        merged = prev_tail * (np.cos(gradation * np.pi * 0.5) ** 2) + now_head * (np.cos((1-gradation) * np.pi * 0.5) ** 2)
        #merged = prev_tail * (1 - gradation) + now_head * gradation
        overlapped = np.append(merged, now_wav[overlap_length:-overlap_length])
        return overlapped
  
if __name__ ==  "__main__":
    root = Tk()
    app = client(master=root)
    while True:
        app.run_application()
        root.update()
    app.mainloop()
    
  