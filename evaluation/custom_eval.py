import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

traceability_data = {
    'Sample Output': [f'Sample {i+1}' for i in range(10)],
    'Directly retrieved': [4, 3, 2.5, 1, 5, 3.5, 2.5, 4, 2.5, 1.5], 
    'Partially inferred': [0, 1, 1, 1, 0, 1, 0.5, 0, 2, 0.5], 
    'Not from RAG (likely hallucinated)': [0, 0, 0.25, 0, 0, 0, 0.5, 0.5, 0, 1.5]  
}
df_traceability = pd.DataFrame(traceability_data).set_index('Sample Output')

personalization_data = {
    'Sample Output': [f'Sample {i+1}' for i in range(10)],
    'Mentions ingredients': ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'], # 100% success
    'Respects allergies': ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'], # 100% success
    'Mentions season': ['1', '0', '0', '0', '1', '0', '0', '0', '1', '0'], # 40% success
    'References user RAG entries': ['1', '1', '0', '1', '1', '1', '1', '0', '1', '1'] # 80% success
}
df_personalization = pd.DataFrame(personalization_data).set_index('Sample Output')

# traceability bar chart
plt.figure(figsize=(12, 7)) # Adjusted figure size for 10 samples
df_traceability.plot(kind='bar', stacked=True, colormap='viridis', ax=plt.gca())
plt.title('Traceability of LLM Suggestions by Sample (10 Samples Throughout 1 Session)', fontsize=16)
plt.xlabel('Sample Output', fontsize=12)
plt.ylabel('Number of Suggestions', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Traceability Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# personalization score table
print("Personalization Score Summary Table")
print("This table summarizes the personalization criteria met for each of the 10 LLM sample outputs, reflecting the success rates from your original 5-sample data.")
print("\nPersonalization Score Status (10 Samples - Original Success Rate):")
print(df_personalization.to_string())
