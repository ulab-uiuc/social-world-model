import './App.css';

function App() {
  const cards = [
    {
      title: "Who will win the president election?",
      options: [
        { label: "A", percentage: "45%" },
        { label: "B", percentage: "40%" },
        { label: "C", percentage: "10%" },
        { label: "D", percentage: "1%" },
        { label: "E", percentage: "2%" },
        { label: "F", percentage: "2%" },
      ],
    },


    {
      title: "Who will win the president election?",
      options: [
        { label: "A", percentage: "45%" },
        { label: "B", percentage: "40%" },
        { label: "C", percentage: "10%" },
        { label: "D", percentage: "5%" },
      ],
    },
    {
      title: "Who will win the president election?",
      options: [
        { label: "A", percentage: "45%" },
        { label: "B", percentage: "40%" },
        { label: "C", percentage: "10%" },
        { label: "D", percentage: "5%" },
      ],
    },
    {
      title: "Who will win the president election?",
      options: [
        { label: "A", percentage: "45%" },
        { label: "B", percentage: "40%" },
        { label: "C", percentage: "10%" },
        { label: "D", percentage: "5%" },
      ],
    },

    {
      title: "Who will win the president election?",
      options: [
        { label: "A", percentage: "45%" },
        { label: "B", percentage: "40%" },
        { label: "C", percentage: "10%" },
        { label: "D", percentage: "5%" },
      ],
    },

    {
      title: "Who will win the president election?",
      options: [
        { label: "A", percentage: "45%" },
        { label: "B", percentage: "40%" },
        { label: "C", percentage: "10%" },
        { label: "D", percentage: "5%" },
      ],
    },

    {
      title: "Who will win the president election?",
      options: [
        { label: "A", percentage: "45%" },
        { label: "B", percentage: "40%" },
        { label: "C", percentage: "10%" },
        { label: "D", percentage: "5%" },
      ],
    },

    {
      title: "Who will win the president election?",
      options: [
        { label: "A", percentage: "45%" },
        { label: "B", percentage: "40%" },
        { label: "C", percentage: "10%" },
        { label: "D", percentage: "5%" },
      ],
    },

    {
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