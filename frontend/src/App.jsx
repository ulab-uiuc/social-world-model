import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useParams } from 'react-router-dom';
import './App.css';
import axios from "axios";

function App() {
  const [cards, setCards] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    axios.get("http://127.0.0.1:5000/api/cards")
      .then((response) => {
        setCards(response.data);
      })
      .catch((error) => console.error("Error fetching cards:", error));
  }, []);

  return (
    <Router>
      <div className="app-container">
        <header className="app-header">
          <h1 className="app-title">Openmarket</h1>
          <div className="horizontal-bar"></div>
        </header>
      <Routes>
        <Route path="/" element={<CardList cards={cards} />} />
        <Route path="/details/:card_id" element={<CardDetails cards={cards} />} />
      </Routes>
      </div>
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

  const [selectedOption, setSelectedOption] = useState(null); // 当前选中的选项
  const [amount, setAmount] = useState(0); // 输入的金额

  if (!card) {
    return <div>Card not found</div>;
  }

  const handleOptionClick = (option) => {
    setSelectedOption(option); // 更新选中的选项
  };

  const handleAmountChange = (e) => {
    setAmount(e.target.value); // 更新金额
  };

  return (
    <div className="details-page">

      <div className="left-card">
        <h2>{card.question}</h2>
        <div className="options-table">
          {card.options.map((option, idx) => (
            <div className="option-row" key={idx}>
              <span className="option-label">{option.option}</span>
              <span className="option-percentage">{option.percentage}%</span>
              <button className="yes-button" onClick={() => handleOptionClick(option)}>
                Yes
              </button>
              <button className="no-button" onClick={() => handleOptionClick(option)}>
                No
              </button>
            </div>
          ))}
        </div>
      </div>


      <div className="right-card">
        <h2>Trade Option</h2>
        {selectedOption ? (
          <div className="trade-module">
            <p>Selected: <strong>{selectedOption.option}</strong></p>
            <p>Chance: <strong>{selectedOption.percentage}%</strong></p>
            <div className="amount-input">
              <label htmlFor="amount">Amount: </label>
              <input
                type="number"
                id="amount"
                value={amount}
                onChange={handleAmountChange}
                min="0"
              />
            </div>
            <button className="trade-button">Buy</button>
            <button className="trade-button">Sell</button>
          </div>
        ) : (
          <p>Please select an option to trade.</p>
        )}
      </div>
    </div>
  );
}

export default App;
