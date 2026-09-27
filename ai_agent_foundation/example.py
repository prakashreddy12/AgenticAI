def understand_request(request):
    return request.lower()

def choose_action(request):
    if "delete" in request:
        return "Ask user for next action"
    return "No further action"

request = understand_request("View the unnecessary drama from life")
action = choose_action(request)
print(action)