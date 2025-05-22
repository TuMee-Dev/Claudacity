mkdir -p ~/.node-certs
cp ~/.mitmproxy/mitmproxy-ca-cert.pem ~/.node-certs/
export HTTP_PROXY=http://localhost:8080
export HTTPS_PROXY=http://localhost:8080
export NODE_EXTRA_CA_CERTS=~/.mitmproxy/mitmproxy-ca-cert.pem
export NO_PROXY=localhost,127.0.0.1,::1
# Try adding this as well
#export NODE_OPTIONS=--use-openssl-ca
claude
