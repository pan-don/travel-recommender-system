document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('search-form');
    const modeSelector = document.getElementById('query_mode');
    const dynamicFields = document.getElementById('dynamic-fields');
    const resultsContainer = document.getElementById('results-container');
    const statsContainer = document.getElementById('query-stats');
    const statTotal = document.getElementById('stat-total');
    const statTime = document.getElementById('stat-time');

    // Global metadata arrays
    let categoriesList = [];
    let citiesList = [];
    let itemsList = [];

    // Fetch dropdown data on load
    fetch('/api/destinations')
        .then(res => res.json())
        .then(data => {
            categoriesList = data.categories || [];
            citiesList = data.cities || [];
            itemsList = data.items || [];
            updateFormFields(); // initialize form with default Q1
        })
        .catch(err => console.error("Error loading destinations:", err));

    // Handle mode change
    modeSelector.addEventListener('change', updateFormFields);

    function updateFormFields() {
        dynamicFields.innerHTML = ''; // clear current
        const mode = modeSelector.value;

        // Helper to append a template
        const appendTemplate = (id) => {
            const template = document.getElementById(id);
            if(template) {
                dynamicFields.appendChild(template.content.cloneNode(true));
            }
        };

        // Determine which fields to show based on mode
        switch(mode) {
            case 'q1': // Semantic
            case 'q7': // PCA
                appendTemplate('field-query-single');
                appendTemplate('field-k');
                break;
            case 'q2': // Category Filter
                appendTemplate('field-query-single');
                appendTemplate('field-category');
                appendTemplate('field-k');
                populateSelect('category', categoriesList);
                break;
            case 'q3': // City Filter
                appendTemplate('field-query-single');
                appendTemplate('field-cities');
                appendTemplate('field-k');
                populateSelect('cities', citiesList);
                break;
            case 'q4': // Broad Semantic (previously rating)
                appendTemplate('field-query-single');
                // Set default K to 50 for broad search implicitly or explicitly
                appendTemplate('field-k');
                setTimeout(() => document.getElementById('k').value = 50, 0);
                break;
            case 'q5': // Item-based
                appendTemplate('field-items');
                appendTemplate('field-k');
                populateItemsSelect('item_id', itemsList);
                break;
            case 'q6': // Ensemble
                appendTemplate('field-query-multi');
                appendTemplate('field-k');
                break;
        }
    }

    function populateSelect(id, list) {
        setTimeout(() => {
            const select = document.getElementById(id);
            if (!select) return;
            select.innerHTML = '';
            list.forEach(item => {
                const opt = document.createElement('option');
                opt.value = item;
                opt.textContent = item;
                select.appendChild(opt);
            });
        }, 0);
    }

    function populateItemsSelect(id, items) {
        setTimeout(() => {
            const select = document.getElementById(id);
            if (!select) return;
            select.innerHTML = '';
            items.slice(0, 100).forEach(item => { // Limit to 100 to avoid freezing
                const opt = document.createElement('option');
                opt.value = item.Place_Id;
                opt.textContent = `${item.Place_Id} - ${item.Place_Name}`;
                select.appendChild(opt);
            });
        }, 0);
    }

    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        resultsContainer.innerHTML = '<div class="text-center py-10"><div class="animate-pulse flex flex-col items-center"><div class="h-8 w-8 bg-blue-400 rounded-full mb-4"></div><p class="text-gray-500">Searching vector space...</p></div></div>';
        statsContainer.classList.add('hidden');

        const mode = modeSelector.value;
        const formData = new FormData(form);

        let endpoint = '';
        let payload = {};
        let method = 'POST';

        // Extract K safely
        const k = parseInt(formData.get('k') || 5);

        // Build request payload based on mode
        switch(mode) {
            case 'q1':
                endpoint = '/api/search';
                payload = { query: formData.get('query'), k: k };
                break;
            case 'q2':
                endpoint = '/api/search/category';
                payload = { query: formData.get('query'), category: formData.get('category'), k: k };
                break;
            case 'q3':
                endpoint = '/api/search/city';
                // Handle multiple select
                const selectElement = document.getElementById('cities');
                const selectedCities = Array.from(selectElement.selectedOptions).map(opt => opt.value);
                payload = { query: formData.get('query'), cities: selectedCities, k: k };
                break;
            case 'q4':
                endpoint = '/api/search/rating'; // Reusing endpoint name but it's broad search
                payload = { query: formData.get('query'), k: k };
                break;
            case 'q5':
                endpoint = `/api/similar/${formData.get('item_id')}?k=${k}`;
                method = 'GET';
                payload = null;
                break;
            case 'q6':
                endpoint = '/api/search/ensemble';
                payload = {
                    queries: [formData.get('query1'), formData.get('query2'), formData.get('query3')],
                    k: k
                };
                break;
            case 'q7':
                endpoint = '/api/search/pca';
                payload = { query: formData.get('query'), k: k };
                break;
        }

        try {
            const options = {
                method: method,
                headers: { 'Content-Type': 'application/json' }
            };
            if (payload && method === 'POST') {
                options.body = JSON.stringify(payload);
            }

            const response = await fetch(endpoint, options);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Unknown error occurred');
            }

            renderResults(data);

        } catch (error) {
            resultsContainer.innerHTML = `
                <div class="bg-red-50 border-l-4 border-red-500 p-4 rounded shadow">
                    <div class="flex">
                        <div class="flex-shrink-0">
                            <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
                            </svg>
                        </div>
                        <div class="ml-3">
                            <h3 class="text-sm font-medium text-red-800">Error executing query</h3>
                            <div class="mt-2 text-sm text-red-700"><p>${error.message}</p></div>
                        </div>
                    </div>
                </div>`;
        }
    });

    function renderResults(data) {
        resultsContainer.innerHTML = '';

        // Update stats
        statTotal.textContent = data.total_found;
        statTime.textContent = data.query_time_ms;
        statsContainer.classList.remove('hidden');

        if (!data.results || data.results.length === 0) {
            resultsContainer.innerHTML = '<div class="bg-white p-8 rounded-lg shadow text-center text-gray-500">No matching destinations found. Try adjusting your query.</div>';
            return;
        }

        data.results.forEach(item => {
            // Convert score to percentage for the bar
            const scorePercent = Math.max(0, Math.min(100, item.score * 100)).toFixed(1);

            // Determine badge color based on category
            let badgeColor = 'bg-blue-100 text-blue-800';
            if(item.category.toLowerCase().includes('budaya')) badgeColor = 'bg-amber-100 text-amber-800';
            else if(item.category.toLowerCase().includes('alam')) badgeColor = 'bg-green-100 text-green-800';
            else if(item.category.toLowerCase().includes('bahari') || item.category.toLowerCase().includes('pantai')) badgeColor = 'bg-cyan-100 text-cyan-800';

            const card = document.createElement('div');
            card.className = "bg-white rounded-lg shadow p-5 flex flex-col md:flex-row gap-4 items-start md:items-center hover:shadow-md transition border border-transparent hover:border-blue-200";
            card.innerHTML = `
                <div class="flex-shrink-0 w-12 h-12 rounded-full bg-blue-600 flex items-center justify-center text-white font-bold text-xl shadow-inner">
                    #${item.rank}
                </div>
                <div class="flex-grow">
                    <div class="flex items-center gap-2 mb-1">
                        <h3 class="text-xl font-bold text-gray-900">${item.name}</h3>
                        <span class="px-2.5 py-0.5 rounded-full text-xs font-medium ${badgeColor}">
                            ${item.category}
                        </span>
                    </div>
                    <div class="text-gray-600 text-sm flex items-center gap-2 mb-3">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
                        ${item.city}
                        <span class="text-gray-400">|</span>
                        <span class="font-mono text-xs">ID: ${item.id}</span>
                    </div>

                    <!-- Score Bar -->
                    <div class="w-full">
                        <div class="flex justify-between text-xs mb-1">
                            <span class="font-medium text-gray-700">Similarity Score</span>
                            <span class="font-mono text-blue-600 font-bold">${item.score.toFixed(4)}</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div class="bg-blue-600 h-2 rounded-full" style="width: ${scorePercent}%"></div>
                        </div>
                    </div>
                </div>
            `;
            resultsContainer.appendChild(card);
        });
    }
});
