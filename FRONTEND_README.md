# RAG-based AI Math Tutor - Frontend

A beautiful, modern web interface for your RAG-based AI Math Tutor that helps students learn geometry and transformations.

## 🚀 Features

- **Modern Chat Interface**: Clean, responsive design with real-time messaging
- **Smart Suggestions**: Pre-built question chips to get started quickly
- **Real-time Responses**: Live typing indicators and smooth animations
- **Mobile Responsive**: Works perfectly on desktop, tablet, and mobile devices
- **Video Integration**: Direct links to relevant Khan Academy videos with timestamps

## 🎨 Design Highlights

- **Gradient Backgrounds**: Beautiful purple-blue gradient design
- **Glass Morphism**: Modern frosted glass effects
- **Smooth Animations**: Hover effects and transitions
- **Typography**: Clean Inter font for excellent readability
- **Color Coding**: Timestamps and video titles are highlighted

## 🛠️ Technical Stack

- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Backend**: Flask (Python)
- **AI**: OpenAI GPT-3.5-turbo + text-embedding-3-small
- **Styling**: Custom CSS with modern design patterns

## 📁 File Structure

```
├── app.py                 # Flask backend server
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── style.css         # Modern CSS styling
│   └── script.js         # Frontend JavaScript
├── process_incoming.py   # Original CLI version
└── embeddings.joblib     # Pre-computed embeddings
```

## 🚀 How to Run

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and go to:
   ```
   http://localhost:8080
   ```

3. **Start chatting** with your AI Math Tutor!

## 💡 Usage

- Type your questions in the input field
- Click the example question chips to get started
- Press Enter to send messages
- Type "bye" to exit (though the web version runs continuously)

## 🎯 Example Questions

- "What is transformation?"
- "Explain similar triangles"
- "How do I reflect over y=x?"
- "What are intersecting chords?"
- "Show me proofs about angles"

## 🔧 Customization

- **Colors**: Modify the CSS variables in `style.css`
- **Questions**: Update the example chips in `index.html`
- **Styling**: Adjust the design in `style.css`
- **Functionality**: Enhance features in `script.js`

## 📱 Responsive Design

The interface automatically adapts to different screen sizes:
- **Desktop**: Full-width layout with side-by-side messages
- **Tablet**: Optimized spacing and sizing
- **Mobile**: Stacked layout with touch-friendly buttons

Enjoy your beautiful AI Math Tutor interface! 🎓✨
