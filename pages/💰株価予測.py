# WebAPIなどの準備
import sys
import numpy as np
import time
import pandas_datareader as pdd
from pandas.core.frame import DataFrame
import pandas as pd
import datetime
import os
import glob
import streamlit as st
from matplotlib import pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pytorch_lightning.loggers import CSVLogger
import torchmetrics
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint

# 標準化関数（StandardScaler）をインポート
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# 入力項目
code_input = st.number_input('銘柄コード',min_value=1000,max_value=9999)
date_input = st.date_input('予測開始日を入力してください',min_value=datetime.datetime(2022,1,1),max_value=datetime.datetime.today())
days_ahead = st.number_input('何日先のデータを予測する？',min_value=1,max_value=10)


if st.button("実行"):
    code_JP = str(code_input) + '.JP' # 銘柄コードの対応
    proba_date = date_input - datetime.timedelta(weeks = 480) # 過去10年分のデータ
    stock_price_proba = pdd.data.DataReader(code_JP, 'stooq', proba_date, date_input) # pandas_datareaderで株価取得

    # 前処理
    df = pd.DataFrame(stock_price_proba).sort_values(['Date']) # 日付データが降順？になっているので直す。
    df = df.reset_index()
    df = df.drop(['Open', 'Low', 'High', 'Volume'], axis=1) # 今回は終値のみのため不要なデータ削除(今後、特徴量として使う)

    train_val = df.iloc[:int(len(df)*0.8),:] # trainデータとvalデータに分割(80%)
    test = df.iloc[int(len(df)*0.8):,:] # testデータに分割(20%)
    proba = df.iloc[int(len(df)-20):,:]

    # 標準化処理
    train_val_close = train_val['Close'].values.reshape(-1,1)
    test_close = test['Close'].values.reshape(-1,1)
    proba_close = proba['Close'].values.reshape(-1,1)

    scaler = StandardScaler() # インスタンス化

    train_val_std = scaler.fit_transform(train_val_close)
    test_std = scaler.fit_transform(test_close)
    proba_std = scaler.fit_transform(proba_close)

    # 入力値、出力値のデータ格納

    input_data = [] # 入力データ(過去20日分の株価)
    output_data = [] # 出力データ(入力データ+1日)

    # リストにデータを格納
    for i in range(len(train_val_std) - 20):
        input_data.append(train_val_std[i:i + 20])
        output_data.append(train_val_std[i + 20])

    input_data_test = []
    output_data_test = []

    for i in range(len(test_std) - 20):
        input_data_test.append(test_std[i:i + 20])
        output_data_test.append(test_std[i + 20])


    input_data_prova = []

    input_data_prova.append(proba_std[len(proba_std)-20:]) 

    # ndarrayに変換
    train_val_input = np.array(input_data)
    train_val_output = np.array(output_data)
    test_input = np.array(input_data_test)
    test_output =np.array(output_data_test)

    # train,val分割
    val_len = int(train_val_input.shape[0]*0.2)
    train_len = int(train_val_input.shape[0] - val_len)

    # 訓練データ
    train_input = train_val_input[:train_len]
    train_output = train_val_output[:train_len]

    # 検証データ
    val_input = train_val_input[:val_len]
    val_output = train_val_output[:val_len]

    # データセット作成
    train_x = torch.Tensor(train_input)
    val_x = torch.Tensor(val_input)
    test_x = torch.Tensor(test_input)
    train_t = torch.Tensor(train_output)
    val_t = torch.Tensor(val_output)
    test_t = torch.Tensor(test_output)

    train_dataset = TensorDataset(train_x, train_t)
    val_dataset = TensorDataset(val_x, val_t)
    test_dataset = TensorDataset(test_x,test_t)

    test_prova = torch.Tensor(input_data_prova)

    pl.seed_everything(0)

    batch_size = len(test) // 50

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size)
    test_loader = DataLoader(test_dataset,batch_size)

    class Net(pl.LightningModule):

        def __init__(self):

            super().__init__()

            self.lstm = nn.LSTM(1,100)
            self.linear = nn.Linear(100, 1)

        def forward(self, x):
            x = x.T.view(x.size(1),x.size(0),1)
            out, (h_n, c_n) = self.lstm(x)
            h = h_n.view(h_n.size(1), -1)
            h = self.linear(h)
            return h

        def training_step(self, batch, batch_idx):
            x, t = batch
            y = self(x)
            loss = F.mse_loss(y, t)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, t = batch
            y = self(x)
            loss = F.mse_loss(y, t)
            self.log('val_loss', loss, on_step=False, on_epoch=True)
            return loss

        def test_step(self, batch, batch_idx):
            x, t = batch
            y = self(x)
            loss = F.mse_loss(y, t)
            self.log('test_loss', loss, on_step=False, on_epoch=True)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=0.03)
            return optimizer

    # 再現性を確保するためシードを固定する
    pl.seed_everything(0)

    # モデルをインスタンス化
    net = Net()

    # モデルのパラメータの保存場所を指定
    checkpoint_callback = ModelCheckpoint(
        dirpath='lightning_logs',
        filename= 'best_prova',
        verbose=False,
        monitor='val_loss',
        mode='min',
    )

    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        patience = 5
    )

    device = torch.device('cpu')

    # 学習の実行
    trainer = pl.Trainer(deterministic=True, max_epochs=1000,callbacks=[checkpoint_callback,early_stopping])
    trainer.fit(net.to(device), train_loader, val_loader)

    state_dict = torch.load(checkpoint_callback.best_model_path)

    trainer.test(dataloaders=test_loader,ckpt_path=checkpoint_callback.best_model_path)

    torch.save(net.state_dict(), 'stock.pt')
    net.load_state_dict(torch.load('stock.pt'))
    net.eval()

    y = net(test_prova)
    y_n = y.to('cpu').detach().numpy().copy()
    result = scaler.inverse_transform(y_n)

    df_result = df.append({'Date':date_input + datetime.timedelta(days=1),'Close':float(result)}, ignore_index=True)
    test_result =test.append({'Date':date_input + datetime.timedelta(days=1),'Close':float(result)}, ignore_index=True)

    fig, ax = plt.subplots()

    ax.plot(test_result['Date'],test_result['Close'])

    st.pyplot(fig)

    date_result = test_result['Date']
    close_result = test_result['Close']

    st.markdown(str(date_result.iloc[-1]) + 'の終値の予測値は' + str(close_result.iloc[-1]) + '円です。')