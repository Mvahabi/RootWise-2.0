from frontend import demo
from logic import initialize_rag

if __name__ == "__main__":
  result = initialize_rag('./system_data')
  print(f"RAG init result: {result}")

  demo.queue()
  demo.launch(server_name="0.0.0.0", server_port=7860)
  print("herelo")