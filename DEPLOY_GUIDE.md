# 🌐 河道水质智能评估平台 - 部署指南

## 方式一：同一 WiFi 访问（最简单）

本机已开启全网监听，同一局域网内设备访问：

```
http://172.20.140.248:8501
```

## 方式二：Streamlit Cloud 部署（推荐，永久在线）

### 第一步：上传到 GitHub

```bash
# 1. 进入项目目录
cd river-water-quality-assessment

# 2. 初始化 git
git init
git add .
git commit -m "水质评估平台完整代码"

# 3. 在 GitHub 新建仓库（网页操作）
#    访问 https://github.com/new
#    仓库名：water-quality-assessment
#    勾选 Public

# 4. 推送代码
git remote add origin https://github.com/你的用户名/water-quality-assessment.git
git branch -M main
git push -u origin main
```

### 第二步：部署到 Streamlit Cloud

1. 访问 https://streamlit.io/cloud
2. 用 GitHub 账号登录
3. 点击 "New app"
4. 选择仓库 `water-quality-assessment`
5. 分支选 `main`
6. 主文件路径填 `app.py`
7. 点击 "Deploy"

几分钟后就能得到公网地址：
```
https://water-quality-assessment.streamlit.app
```

## 方式三：本地 ngrok 隧道

如需临时公网地址，在本机终端执行：

```bash
# 安装 ngrok
# 访问 https://ngrok.com/download 下载

# 启动隧道
ngrok http 8501
```

得到类似 `https://xxxx.ngrok.io` 的地址。

---

## 启动命令（本机）

```bash
cd river-water-quality-assessment
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
