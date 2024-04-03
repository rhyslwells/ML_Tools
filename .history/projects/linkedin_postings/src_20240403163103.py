

# function to give scatter for each elbow number
def scatter_elbow(X, elbow_num, var1, var2):
    #elbow plot for kmeans clustering
    """
    Apply clustering with elbow method and plot a scatter plot with cluster information.

    Parameters:
    - X: DataFrame, input data for clustering
    - elbow_num: int, number of clusters determined by elbow method
    - var1, var2: str, names of the variables for the scatter plot

    Returns:
    None (plots the scatter plot)
    """
    # Apply clustering with elbow number
    kmeans = KMeans(elbow_num)
    kmeans.fit(X)

    # Add cluster information
    identified_clusters = kmeans.fit_predict(X)
    X['Cluster'] = identified_clusters

    # Plot
    plt.scatter(X[var1], X[var2], c=X['Cluster'], cmap='rainbow')
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.title(f"{elbow_num}-Clustering for {var1}-{var2}")
    plt.show()

# Example usage:
# scatter_elbow(data, elbow_num, 'var1', 'var2')
    


def pull_company_data(company_id):
    # Read employee counts data
    employee_counts = pd.read_csv('company_details/employee_counts.csv')
    employee_counts = employee_counts[employee_counts['company_id'] == company_id]

    # Read company specialties data
    company_specialties = pd.read_csv('company_details/company_specialties.csv')
    company_specialties = company_specialties[company_specialties['company_id'] == company_id]

    # Read company industries data
    company_industries = pd.read_csv('company_details/company_industries.csv')
    company_industries = company_industries[company_industries['company_id'] == company_id]

    # Read companies data
    companies = pd.read_csv('company_details/companies.csv')
    company_info = companies[companies['company_id'] == company_id]

    # Concatenate all dataframes into one
    processed_data = pd.concat([employee_counts, company_specialties, company_industries, company_info], axis=1)

    return processed_data

# Example usage:
# company_id = 12345  # Replace with the desired company ID
# company_data = pull_company_data(company_id)
# print(company_data)


def pull_job_data(job_id):
    #get all the data for a specific job id

    # Read benefits data
    benefits = pd.read_csv('job_details/benefits.csv')
    benefits = benefits[benefits['job_id'] == job_id]

    # Read job industries data
    job_industries = pd.read_csv('job_details/job_industries.csv')
    job_industries = job_industries[job_industries['job_id'] == job_id]

    # Map industry IDs to industry names using industries.csv
    industries = pd.read_csv('maps/industries.csv')
    job_industries = pd.merge(job_industries, industries, on='industry_id', how='left')

    # Read job skills data
    job_skills = pd.read_csv('job_details/job_skills.csv')
    job_skills = job_skills[job_skills['job_id'] == job_id]

    # Map skill abbreviations to skill names using skills.csv
    skills = pd.read_csv('maps/skills.csv')
    job_skills = pd.merge(job_skills, skills, left_on='skill_abr', right_on='skill_abr', how='left')

    # Read salaries data
    salaries = pd.read_csv('job_details/salaries.csv')
    salaries = salaries[salaries['job_id'] == job_id]

    # Concatenate all dataframes into one
    processed_data = pd.concat([benefits, job_industries, job_skills, salaries], axis=1)

    return processed_data


# Example usage:
# job_id = 12345  # Replace with the desired job ID
# job_data = pull_job_data(job_id)
# print(job_data)
