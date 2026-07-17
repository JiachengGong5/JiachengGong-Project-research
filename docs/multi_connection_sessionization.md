# Activity Trace Reconstruction

Tracking related connections is part of the project, but it belongs to the
interpretation layer rather than the LSTM input layer.

The main pipeline is:

```text
PCAP
-> Zeek protocol logs
-> chronological event sequence
-> LSTM classification
-> salient event span
-> Zeek-based activity trace reconstruction
```

The model still receives only protocol-semantic tokens such as protocol,
service, connection state, DNS result, HTTP method, TLS state, and Zeek
anomaly name. It does not receive IP addresses, Zeek `uid`, byte counts,
duration, packet totals, rates, means, standard deviations, or rolling-window
statistics.

After the model identifies an important span, the project maps that span back
to Zeek logs using timestamp ranges and Zeek `uid` values. This can connect
related records across logs:

- `conn.log` for connection state and service context,
- `dns.log` for query and response behavior,
- `http.log` for request method and response status,
- `ssl.log` for TLS state,
- `weird.log` for protocol anomalies.

This means the final explanation can describe the activity trace instead of
only reporting a class label. For example, a salient Recon span can be traced
back to repeated connection attempts across ports, while a Web-based span can
be traced back to the related connection and HTTP records.

The important boundary is:

- `uid`, timestamps, IP addresses, and raw Zeek records may be used after
  prediction to explain and reconstruct the activity.
- Those tracking fields are not added to the LSTM training tokens, so they do
  not become leakage-prone model features.

If later analysis shows that a class needs stronger grouping than a simple
time-window and `uid` trace, that grouping should be documented as an
interpretability improvement or ablation, not as a replacement for the main
chronological event-sequence model.
