import React from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useParams } from 'react-router-dom';
import './App.css';

function App() {
  const cards = [
    {
      id: 1,
      title: "Who will win the president election?",
      options: [
        { label: "A", percentage: "45%" },
        { label: "B", percentage: "40%" },
        { label: "C", percentage: "10%" },
        { label: "D", percentage: "5%" },
      ],
    },
    {
      id: 2,
      title: "Who will win the president election?",
      options: [
        { label: "A", percentage: "45%" },
        { label: "B", percentage: "40%" },
        { label: "C", percentage: "10%" },
        { label: "D", percentage: "5%" },
      ],
    },
    {
      id: 3,
      title: "Who will win the president election?",
      options: [
        { label: "A", percentage: "45%" },
        { label: "B", percentage: "40%" },
        { label: "C", percentage: "10%" },
        { label: "D", percentage: "5%" },
      ],
    },
  ];

  return (
    <Router>
      <Routes>
        <Route path="/" element={<CardList cards={cards} />} />
        <Route path="/details/:id" element={<CardDetails cards={cards} />} />
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
          key={card.id}
          onClick={() => navigate(`/details/${card.id}`)}
        >
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

function CardDetails({ cards }) {
  const { id } = useParams();
  const card = cards.find((c) => c.id === parseInt(id));

  if (!card) {
    return <div>Card not found</div>;
  }

  return (
    <div className="details-container">
      <h2>{card.title}</h2>
      <div className="options-table">
        {card.options.map((option, idx) => (
          <div className="option-row" key={idx}>
            <span className="option-label">{option.label}</span>
            <span className="option-percentage">{option.percentage}</span>
            <button className="yes-button">Yes</button>
            <button className="no-button">No</button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
