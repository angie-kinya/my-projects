import React, { useState, useEffect } from 'react';
import axios from 'axios';

function FitnessPlan() {
    const [plan, setPlan] = useState(null);

    useEffect(() => {
        axios.get('/plans/get-plan/')
            .then(response => {
                setPlan(response.data);
            })
            .catch(error => {
                console.error('There was an error fetching the data!', error);
            });
    }, []);

    return (
        <div>
            <h1>Personalized Fitness Plan</h1>
            {plan ? <p>{JSON.stringify(plan)}</p> : <p>Loading...</p>}
        </div>
    );
}

export default FitnessPlan;