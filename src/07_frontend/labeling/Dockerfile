FROM heartexlabs/label-studio:latest

# Prevent apt from showing prompts
ENV LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK="true"
ENV LABEL_STUDIO_USERNAME="artexmg@gmail.com"
ENV LABEL_STUDIO_PASSWORD="secure"
# ENV GOOGLE_APPLICATION_CREDENTIALS="/secrets/ac215-dermaid.json"
ENV GCP_PROJECT="amg-sandbox-115"
ENV GCP_ZONE="us-central1-a"
# ENV LABEL_STUDIO_HOST="http://35.239.206.224.sslip.io/labeling/"

# Expose port of API service
EXPOSE 8080

USER root

# RUN mkdir /secrets

ADD docker-volumes/label-studio /label-studio/data
# ADD ../../secrets/ac215-dermaid.json /secrets/

# Entry point
# ENTRYPOINT ["/bin/bash","./docker-entrypoint.sh"]