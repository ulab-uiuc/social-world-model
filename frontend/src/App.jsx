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
    <Router>
      <Routes>
        <Route path="/" element={<CardList cards={cards} />} />
        <Route path="/details/:card_id" element={<CardDetails cards={cards} />} />
      </Routes>
    </Router>
  );
}

function CardList({ cards }) {
  const navigate = useNavigate();

  return (
    <div className="card-container">
      {cards.map((card) => (
        <div
          className="card"
          key={card.card_id}
          onClick={() => navigate(`/details/${card.card_id}`)}
        >
          <h3 className="card-title">{card.question}</h3>
          <div className="menu">
            {card.options.map((option, idx) => (
              <div className="menu-item" key={idx}>
                <span className="option-label">{option.option}</span>
                <span className="option-percentage">{option.percentage}%</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function CardDetails({ cards }) {
  const { card_id } = useParams();
  const card = cards.find((c) => c.card_id === card_id);

  if (!card) {
    return <div>Card not found</div>;
  }

  return (
    <div className="details-container">
      <h2>{card.question}</h2>
      <div className="options-table">
        {card.options.map((option, idx) => (
          <div className="option-row" key={idx}>
            <span className="option-label">{option.option}</span>
            <span className="option-percentage">{option.percentage}%</span>
            <button className="yes-button">Yes</button>
            <button className="no-button">No</button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
