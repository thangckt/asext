// Source: https://github.com/Matti-Krebelder/Website-IP-Logger
// Thang modifications:
// - With the help from GPT
// - Change from using `sendToDiscord` to `sendDataToGoogleApp`
// ref: Using Google App Mail: https://github.com/dwyl/learn-to-send-email-via-google-script-html-no-server

(function () {

    // Array of API endpoints with their respective data extraction logic
    const APIs = [
        {
            url: 'https://ipapi.co/json',
            parse: (data) => ({
                ip: data.ip,
                org: data.org,
                city: data.city,
                region: data.region,
                country: data.country_name,
                postal: data.postal,
                asn: data.asn,
                latitude: data.latitude,
                longitude: data.longitude,
            }),
        },
        {
            url: 'https://ipinfo.io/json',
            parse: (data) => ({
                ip: data.ip,
                org: data.org?.split(' ').slice(1).join(" ") ?? 'Unk',
                city: data.city,
                region: data.region,
                country: data.country,
                postal: data.postal,
                asn: data.org?.split(' ')[0] ?? 'Unk',
                latitude: data.loc?.split(',')[0],
                longitude: data.loc?.split(',')[1],
            }),
        },
        {
            url: 'https://api.ip.sb/geoip',
            parse: (data) => ({
                ip: data.ip,
                org: data.organization,
                city: data.city,
                region: data.region,
                country: data.country,
                postal: data.postal_code,
                asn: data.asn,
                latitude: data.latitude,
                longitude: data.longitude,
            }),
        },
        {
            url: 'https://ipwho.is',
            parse: (data) => ({
                ip: data.ip,
                org: data.connection?.isp,
                city: data.city,
                region: data.region,
                country: data.country,
                postal: data.postal,
                asn: data.connection?.asn,
                latitude: data.latitude,
                longitude: data.longitude,
            }),
        },
        {
            url: 'https://api.techniknews.net/ipgeo',
            parse: (data) => ({
                ip: data.ip,
                org: data.isp,
                city: data.city,
                region: data.region,
                country: data.country,
                postal: data.zip,
                asn: data.as.split(' ')[0],
                latitude: data.lat,
                longitude: data.lon,
            }),
        },
    ];

    function isEmpty(obj) {
        return Object.keys(obj).length === 0;
    }

    // Fetch visitor info using multiple IP APIs with fallback
    async function getVisitorInfo() {
        let visitorInfo = {};

        // Iterate over the APIs until one returns valid data
        for (const api of APIs) {
            try {
                const res = await fetch(api.url);
                const jdata = await res.json();
                if (jdata && jdata.ip) {
                    visitorInfo = api.parse(jdata);
                    if (!isEmpty(visitorInfo)) {
                        break; // Stop if valid data is obtained
                    }
                }
            } catch (error) {
                console.error(`Failed to get visitor info from ${api.url}:`, error);
            }
        }

        // Default if all attempts fail
        if (isEmpty(visitorInfo)) {
            visitorInfo = {
                org: 'not available',
            };
        }

        return visitorInfo;
    }

    // Get browser information from user agent string
    function getBrowserInfo() {
        const ua = navigator.userAgent;
        const info = {
            browser: 'Unk',
            os: 'Unk',
            arch: 'Unk',
            device: 'Unk',
            screen: `${window.screen.width}x${window.screen.height} @${window.devicePixelRatio}x`,
            language: navigator.language || 'Unk',
            isTouch: 'ontouchstart' in window || navigator.maxTouchPoints > 0
        };

        // --- Arch detection ---
        let arch = navigator.userAgentData?.architecture || navigator.platform || 'Unknown';
        if (arch.startsWith('Linux ')) {
            arch = arch.replace('Linux ', '');
        }
        info.arch = arch;

        // --- Browser detection (same as before) ---
        if (/iP(hone|od|ad)/.test(ua)) {
            if (/Safari/.test(ua) && !/CriOS/.test(ua) && !/FxiOS/.test(ua)) {
                const version = ua.match(/Version\/(\d+\.\d+)/)?.[1] || 'Unk';
                info.browser = `Safari-${version}`;
            } else if (/CriOS/.test(ua)) {
                info.browser = `Chrome-${ua.match(/CriOS\/(\d+\.\d+)/)?.[1] || 'Unk'}`;
            } else if (/FxiOS/.test(ua)) {
                info.browser = `Firefox-${ua.match(/FxiOS\/(\d+\.\d+)/)?.[1] || 'Unk'}`;
            } else {
                info.browser = 'WebKit-based-iOS';
            }
        } else {
            const browserData = ua.match(/(opera|chrome|safari|firefox|msie|trident(?=\/))\/?\s*(\d+)/i) || [];
            if (/trident/i.test(browserData[1])) {
                const version = (/\brv[ :]+(\d+)/g.exec(ua) || [])[1] || 'Unk';
                info.browser = `Internet Explorer-${version}`;
            } else if (browserData[1] === 'Chrome') {
                const temp = ua.match(/\b(OPR|Edg)\/(\d+)/);
                if (temp) {
                    info.browser = `${temp[1] === 'OPR' ? 'Opera' : 'Edge'}-${temp[2]}`;
                } else {
                    info.browser = `Chrome-${browserData[2]}`;
                }
            } else if (browserData[1]) {
                info.browser = `${browserData[1]}-${browserData[2]}`;
            }
            if (/Safari/.test(ua) && !/Chrome/.test(ua)) {
                const version = ua.match(/Version\/(\d+\.\d+)/)?.[1] || 'Unk';
                info.browser = `Safari-${version}`;
            }
        }

        // --- OS detection (same as before) ---
        if (/Windows NT/.test(ua)) {
            info.os = `Windows-${ua.match(/Windows NT (\d+\.\d+)/)?.[1] || 'Unk'}`;
        } else if (/Mac OS X/.test(ua)) {
            info.os = `macOS-${ua.match(/Mac OS X (\d+[_\.\d]+)/)?.[1].replace(/_/g, '.') || 'Unk'}`;
        } else if (/Android/.test(ua)) {
            info.os = `Android-${ua.match(/Android (\d+(\.\d+)?)/)?.[1] || 'Unk'}`;
        } else if (/Linux/.test(ua)) {
            let distro = 'Linux';
            let version = 'Unk';

            if (/Ubuntu/i.test(ua)) distro = 'Ubuntu';
            else if (/Fedora/i.test(ua)) distro = 'Fedora';
            else if (/Arch/i.test(ua)) distro = 'Arch';
            else if (/Debian/i.test(ua)) distro = 'Debian';

            const versionMatch = ua.match(/(Ubuntu|Fedora|Debian)\/?(\d+[\.\d]*)/i);
            if (versionMatch) version = versionMatch[2];

            info.os = `${distro}-${version}`;
        } else if (/iP(hone|od|ad)/.test(ua)) {
            info.os = 'iOS';
        }

        // --- Device type detection ---
        if (/Mobi|iPhone|Android.+Mobile|Windows Phone/i.test(ua)) {
            info.device = 'Mobile';
        } else if (/iPad|Tablet|Nexus 7|SM-T|Kindle|Silk/i.test(ua)) {
            info.device = 'Tablet';
        } else if (/Windows|Macintosh|X11|Linux/i.test(ua)) {
            info.device = 'Desktop';
        } else {
            info.device = 'Unk';
        }

        return info;
    }

    // Get current timestamp
    function getTimestamp() {
        const parts = new Intl.DateTimeFormat("en-US", {
            timeZone: "Asia/Seoul",
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false // 24-hour format
        }).formatToParts(new Date());
        const get = (type) => parts.find(p => p.type === type)?.value ?? 'Unk';
        return `${get('month')}/${get('day')}/${get('year')} ${get('hour')}:${get('minute')}:${get('second')}`;
    }

    // Async function to send JSON data to Google Sheets via Google Apps Script
    const ScriptId_visitorSheet = 'AKfycbzM04ouw1vGf5wOZs4106A95PUbfpahtJ-_7cOl1_vWFGw5xey4YLENbGbiyIgs0Xd2tw'
    const ScriptId_otherSheet = "define_your_own_script_id_here" // Placeholder for another Google Apps Script

    async function sendDataToGoogleApp(jsonData, ScriptId) {
        const URL = `https://script.google.com/macros/s/${ScriptId}/exec`;
        try {
            await fetch(URL, {
                method: 'POST',
                body: JSON.stringify(jsonData),
                headers: { 'Content-Type': "text/plain;charset=utf-8" },
                redirect: 'follow',
            });
        } catch (error) {
            console.error('Error while sending data to Google App:', error);
        }
    }

    // Filter visitors based on IP, ASN, or other fields
    const blackList1 = [ // Define blacklist list-of-dictionaries
        { ip: '114.70.12.225', browser: 'Chrome-145', os: 'Linux-Unk' }, // CANLab desktop
        { ip: '180.228.220.66', browser: 'Chrome-145', os: 'Linux-Unk' }, // Home laptop
        { ip: '180.228.220.66', browser: 'Chrome-145', os: 'Android-10' }, // Home mobile
    ];

    function checkIfBlocked(visitorInfo, blackList) {
        // Check if the visitor is in the blacklist
        for (const blockedInfo of blackList) {
            // Block only if ALL keys in the entry match (AND logic)
            const allMatch = Object.keys(blockedInfo).every(key => {
                const actual = visitorInfo[key]?.toString().trim();
                const expected = blockedInfo[key]?.toString().trim();
                return actual !== undefined && actual.includes(expected);
            });
            if (allMatch) return true;
        }
        return false;
    }


    // Log visitor information and send to Google Sheet
    async function logVisitor() {
        const timestamp = getTimestamp()
        const browserInfo = getBrowserInfo();
        const currentUrl = window.location.href.replace(window.location.origin, '');
        const visitorInfo = await getVisitorInfo();

        const jsonData = {
            timestamp: timestamp,
            ip: visitorInfo.ip,
            org: visitorInfo.org,
            city: visitorInfo.city,
            region: visitorInfo.region,
            country: visitorInfo.country,
            postal: visitorInfo.postal,
            asn: visitorInfo.asn,
            // latitude: visitorInfo.latitude,
            // longitude: visitorInfo.longitude,
            // Browser Info
            browser: browserInfo.browser,
            os: browserInfo.os,
            device: browserInfo.device,
            arch: browserInfo.arch,
            screen: browserInfo.screen,
            language: browserInfo.language,
            isTouch: browserInfo.isTouch,
            // Page Info
            currentUrl: currentUrl,
        };

        // Filter visitors based on the blacklist
        if (checkIfBlocked(jsonData, blackList1)) {
            return; // Do not log or send data for blocked visitors
        }

        await sendDataToGoogleApp(jsonData, ScriptId_visitorSheet);
    }

    // Function trigger the visitor logging when the page loads
    window.addEventListener('load', function () {
        setTimeout(logVisitor, 3000); // Wait for 3000 milliseconds (3 seconds) before calling logVisitor
    });

})();
