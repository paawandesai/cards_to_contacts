# 🪪 AI Business Card Reader

A powerful Streamlit application that uses GPT Vision to extract and organize contact information from business card images. Built for public use - anyone can use it with their own OpenAI API key.

## ✨ Features

- **🔍 Smart Extraction**: Uses GPT-4o/GPT-4o-mini to extract contact information from business cards
- **📊 Multi-card Support**: Process multiple business cards from a single image
- **💰 Cost Tracking**: Real-time cost tracking with transparent pricing
- **📱 Mobile-Friendly**: Responsive design that works on all devices
- **🗂️ Notion Integration**: Optional direct upload to your Notion databases
- **📄 Export Options**: Download results as CSV or Excel files
- **🔒 Privacy First**: No data stored permanently - only in your browser session
- **🎯 Duplicate Detection**: Automatically identifies potential duplicate entries
- **✅ Data Validation**: Built-in validation with confidence scores

## 🚀 Live Demo

🌐 **[Try the app live on Streamlit Cloud](https://your-app-url.streamlit.app)** (Replace with actual URL after deployment)

## 🛠️ Getting Started

### Prerequisites

- OpenAI API key (get one at [OpenAI Platform](https://platform.openai.com/api-keys))
- (Optional) Notion integration for contact management

### Using the Deployed App

1. **Visit the live demo** at the URL above
2. **Enter your OpenAI API key** in the sidebar
3. **Upload business card images** (PNG, JPG, JPEG)
4. **Select your preferred model** (GPT-4o-mini for cost-effectiveness, GPT-4o for best accuracy)
5. **Click "Extract Business Cards"** to process
6. **Review and edit** the extracted data
7. **Export** as CSV/Excel or send to Notion

### Running Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-business-card-reader.git
cd ai-business-card-reader

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

## 📋 API Key Setup

### OpenAI API Key (Required)

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in to your account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-proj-...`)
5. Paste it in the sidebar when using the app

### Notion Integration (Optional)

1. Go to [Notion Integrations](https://www.notion.so/my-integrations)
2. Click "New integration"
3. Give it a name and select your workspace
4. Copy the "Integration Token" (starts with `secret_...`)
5. Create a database in Notion for your contacts
6. Share the database with your integration
7. Copy the database ID from the URL
8. Enter both in the app's sidebar

## 💰 Pricing

The app uses OpenAI's pricing model:

| Model | Cost per Image | Best For |
|-------|---------------|----------|
| GPT-4o-mini | $0.00764 | Cost-effective extraction |
| GPT-4o | $0.01528 | Highest accuracy |

*Additional token costs apply based on processing complexity*

## 🏗️ Project Structure

```
ai-business-card-reader/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── config/
│   ├── __init__.py
│   └── models.py          # GPT model configurations
├── utils/
│   ├── __init__.py
│   ├── gpt_vision.py      # GPT Vision API integration
│   ├── data_processing.py # Data extraction and processing
│   └── notion_client.py   # Notion API integration
└── README.md              # This file
```

## 🔧 Technical Details

### Models Supported
- **GPT-4o-mini**: Fast and cost-effective
- **GPT-4o**: Premium accuracy and performance

### Image Formats
- PNG, JPG, JPEG
- Multiple images supported
- Automatic image optimization

### Data Fields Extracted
- Name, Title, Company
- Email, Phone, Website
- Address, LinkedIn
- Additional notes and social media

### Export Formats
- **CSV**: Simple comma-separated values
- **Excel**: Multi-sheet with summary statistics
- **Notion**: Direct database integration

## 🔒 Privacy & Security

- **No permanent storage**: All data is session-based
- **API keys**: Stored only in browser session
- **No server-side data**: Everything processed client-side
- **Secure transmission**: HTTPS-only communication
- **Open source**: Full code transparency

## 📱 Mobile Support

The app is fully responsive and works on:
- 📱 Mobile phones
- 📱 Tablets
- 💻 Desktop computers
- 🖥️ Large screens

## 🤝 Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Issues & Support

If you encounter any issues:

1. Check the [Issues](https://github.com/yourusername/ai-business-card-reader/issues) page
2. Create a new issue with details
3. Include screenshots if helpful

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Select the forked repository
5. Set main file to `main.py`
6. Deploy!

### Deploy to Other Platforms

The app can be deployed to:
- **Heroku**: Add `setup.sh` and `Procfile`
- **Railway**: Direct GitHub integration
- **Render**: Streamlit-compatible deployment
- **Google Cloud Run**: Containerized deployment

## 📊 Usage Statistics

*Real-time usage metrics coming soon*

## 🆕 Changelog

### Version 1.0.0
- Initial release
- GPT Vision integration
- Notion connectivity
- CSV/Excel export
- Mobile-responsive design

## 🙏 Acknowledgments

- OpenAI for GPT Vision API
- Streamlit for the amazing framework
- Notion for database integration
- Contributors and users

---

**Built with ❤️ using Streamlit and GPT Vision**

*Made for the community - use responsibly and enjoy!*