# Deployment Guide for Backend and Frontend on EC2

This guide outlines the steps to deploy both the backend (Flask + MongoDB) and frontend (React) applications on an EC2 instance.

## 1. Backend Deployment

### 1.1 Connect to EC2 Instance
1. Use SSH to connect to your EC2 instance:
   ```bash
   ssh -i "your-key.pem" ec2-user@your-ec2-ip
   ```

### 1.2 Install Dependencies
#### 1.2.1 Update System
```bash
sudo yum update -y
```

#### 1.2.2 Install Python 3.11
```bash
sudo amazon-linux-extras enable python3.11
sudo yum install -y python3.11
```

#### 1.2.3 Install pip and Poetry
```bash
python3.11 -m ensurepip --upgrade
python3.11 -m pip install --upgrade pip
curl -sSL https://install.python-poetry.org | python3.11 -
```

#### 1.2.4 Configure Poetry
```bash
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 1.3 Install MongoDB
#### 1.3.1 Add MongoDB Official Repository
```bash
echo "[mongodb-org-5.0]
name=MongoDB Repository
baseurl=https://repo.mongodb.org/yum/amazon/2/mongodb-org/5.0/x86_64/
gpgcheck=1
enabled=1
gpgkey=https://www.mongodb.org/static/pgp/server-5.0.asc" | sudo tee /etc/yum.repos.d/mongodb-org-5.0.repo
```

#### 1.3.2 Install MongoDB
```bash
sudo yum install -y mongodb-org
```

#### 1.3.3 Start and Enable MongoDB Service
```bash
sudo systemctl start mongod
sudo systemctl enable mongod
```

#### 1.3.4 Test MongoDB Connection
```bash
mongo --eval 'db.runCommand({ connectionStatus: 1 })'
```

### 1.4 Deploy Backend Code
#### 1.4.1 Clone the Backend Repository
```bash
cd ~
git clone https://github.com/ulab-uiuc/social-world-model.git
cd social-world-model
```

#### 1.4.2 Install Python Dependencies
```bash
poetry env use python3.11
poetry install
```

#### 1.4.3 Initialize Database with Test Data
```bash
cd ~/social-world-model/backend
poetry run python init_db.py
```

#### 1.4.4 Start Flask Backend with Gunicorn
```bash
poetry run gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

---

## 2. Frontend Deployment

### 2.1 Install Node.js
```bash
curl -sL https://rpm.nodesource.com/setup_16.x | sudo bash -
sudo dnf install -y nodejs
```

### 2.2 Deploy Frontend Code
#### 2.2.1 Clone the Frontend Repository
```bash
cd ~
git clone https://github.com/your-repo/frontend.git
cd frontend
```

#### 2.2.2 Install Frontend Dependencies
```bash
npm install
```

#### 2.2.3 Modify API URL to Use EC2 Public IP
Edit `frontend/src/App.jsx` and replace `http://127.0.0.1:5000` with your EC2 public IP.

#### 2.2.4 Build and Serve Frontend
```bash
npm run build
```

---

### 2.3 Deploy Frontend with Nginx
#### 2.3.1 Install Nginx
```bash
sudo dnf install nginx -y
```

#### 2.3.2 Configure Nginx to Serve Frontend
Open the Nginx config file:
```bash
sudo nano /etc/nginx/nginx.conf
```

Add the following inside the `server` block:
```nginx
server {
    listen 80;
    server_name your-ec2-ip;

    location / {
        root /home/ec2-user/frontend/build;
        index index.html;
        try_files $uri /index.html;
    }
}
```

#### 2.3.3 Restart and Enable Nginx
```bash
sudo systemctl restart nginx
sudo systemctl enable nginx
```
