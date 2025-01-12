import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useParams } from 'react-router-dom';
import './App.css';
import axios from "axios";

function App() {
  const [cards, setCards] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:5000/api/cards")
      .then((response) => {
        setCards(response.data); 
      })
      .catch((error) => console.error("Error fetching cards:", error));
  }, []);



  return (
    <div className="card-container">
      {cards.map((card, index) => (
        <div className="card" key={index}>
          <h3 className="card-title">{card.title}</h3>
          <div className="menu">
            {card.options.map((option, idx) => (
              <div className="menu-item" key={idx}>
                <span className="option-label">{option.label}</span>
                <span className="option-percentage">{option.percentage}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export default App;
