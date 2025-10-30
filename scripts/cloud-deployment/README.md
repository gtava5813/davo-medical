# DAVO Medical - Cloud Deployment Guide

## Prerequisites

1. ✅ GitHub repository pushed: https://github.com/gtava5813/davo-medical
2. ✅ DigitalOcean account with $200+ credit
3. ✅ PhysioNet MIMIC-III access
4. ✅ HuggingFace account (for Llama models)

## Step 1: Create H100×8 Droplet

1. Go to: https://cloud.digitalocean.com/droplets/new
2. Choose **GPU Droplet**
3. Select: **NVIDIA H100×8**
   - 8 GPUs
   - 640 GB GPU Memory
   - 1,920 GiB RAM
   - 160 vCPUs
4. Choose region: **NYC3** or **SFO3**
5. Authentication: **SSH Key** (add your public key)
6. Hostname: `davo-h100-training`
7. Click **Create Droplet** ($23.92/hour)

## Step 2: SSH into Droplet


Get droplet IP from DigitalOcean dashboard
ssh root@YOUR_DROPLET_IP

text

## Step 3: Configure Credentials

Download deployment script
wget https://raw.githubusercontent.com/gtava5813/davo-medical/main/scripts/cloud-deployment/deploy_h100.sh

Edit with your credentials
nano deploy_h100.sh

Update these lines:
PHYSIONET_USER="eyelid"
PHYSIONET_PASS="your_actual_password"
HF_TOKEN="your_huggingface_token"
text

## Step 4: Run Deployment

Make executable
chmod +x deploy_h100.sh

Run in tmux (so it continues if disconnected)
tmux new-session -s davo-training

Start deployment
./deploy_h100.sh

Detach from tmux: Ctrl+B, then D
Reattach: tmux attach -t davo-training
text

## Step 5: Monitor Training

Check progress
tail -f /mnt/scratch/davo/train_*.log

GPU usage
watch -n 1 nvidia-smi

Estimated time: 3-4 hours
text

## Step 6: Download Trained Models

On your local machine
scp root@YOUR_DROPLET_IP:/mnt/scratch/davo/davo_trained_models.tar.gz ~/

Extract
cd ~/davo-medical
tar -xzf ~/davo_trained_models.tar.gz

Models are in: models/
text

## Step 7: Destroy Droplet

**IMPORTANT**: Destroy droplet to stop billing!

On DigitalOcean dashboard
Droplets → davo-h100-training → Destroy
text

## Timeline & Cost

| Phase | Duration | Cost |
|-------|----------|------|
| Setup | 10 min | $4 |
| Data Processing | 30 min | $12 |
| Model Training | 3.5 hrs | $250 |
| Packaging | 5 min | $2 |
| **Total** | **4.1 hrs** | **$268** |

## Troubleshooting

### Authentication Errors
- Verify PhysioNet credentials
- Check HuggingFace token

### Out of Memory
- Reduce `BATCH_SIZE` to 8

### GPU Not Detected
- Ensure you selected H100×8 droplet

## Support

Issues? Open a ticket: https://github.com/gtava5813/davo-medical/issues
