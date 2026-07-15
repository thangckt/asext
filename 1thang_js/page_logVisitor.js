// Source: https://github.com/Matti-Krebelder/Website-IP-Logger
// Thang modifications:
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
        const uaData = navigator.userAgentData;

        const info = {
            browser: 'Unk',
            os: 'Unk',
            arch: 'Unk',
            device: 'Unk',
            screen: `${window.screen.width}x${window.screen.height} @${window.devicePixelRatio}x`,
            language: navigator.language || 'Unk',
            isTouch: 'ontouchstart' in window || navigator.maxTouchPoints > 0
        };

        // --- Arch detection (Note: high-entropy hints like architecture require an async call; fallback to platform synchronously) ---
        let arch = navigator.platform || 'Unknown';
        if (arch.startsWith('Linux ')) arch = arch.replace('Linux ', '');
        info.arch = arch;

        // --- Browser detection: Client Hints first (Chromium-based browsers) ---
        if (uaData && Array.isArray(uaData.brands) && uaData.brands.length > 0) {
            // 1. Filter out known GREASE pseudo-brands
            const realBrands = uaData.brands.filter(b =>
                !/not[\s._-]?a[\s._-]?brand/i.test(b.brand) &&
                !/grease/i.test(b.brand)
            );

            // 2. Look for specific distinct brands first (Edge, Opera, Brave, Chrome)
            let primaryBrand = realBrands.find(b => /edg|opera|brave|chrome/i.test(b.brand));

            // 3. Fallback to Chromium if no explicit brand matches, or the remaining generic choice
            if (!primaryBrand) {
                primaryBrand = realBrands.find(b => /chromium/i.test(b.brand)) || realBrands[0];
            }

            if (primaryBrand) {
                // Clean up Edge naming convention variations
                const name = /^edg$/i.test(primaryBrand.brand) ? 'Edge' : primaryBrand.brand;
                info.browser = `${name}-${primaryBrand.version}`;
            }
        }

        // --- Browser detection: UA string fallback ---
        if (info.browser === 'Unk' || info.browser.startsWith('Chromium')) {
            let detectedBrowser = 'Unk';

            if (/iP(hone|od|ad)/.test(ua)) {
                if (/Safari/.test(ua) && !/CriOS/.test(ua) && !/FxiOS/.test(ua)) {
                    detectedBrowser = `Safari-${ua.match(/Version\/(\d+\.\d+)/)?.[1] || 'Unk'}`;
                } else if (/CriOS/.test(ua)) {
                    detectedBrowser = `Chrome-${ua.match(/CriOS\/(\d+\.\d+)/)?.[1] || 'Unk'}`;
                } else if (/FxiOS/.test(ua)) {
                    detectedBrowser = `Firefox-${ua.match(/FxiOS\/(\d+\.\d+)/)?.[1] || 'Unk'}`;
                } else {
                    detectedBrowser = 'WebKit-based-iOS';
                }
            } else {
                // Refined Regex sequence matching
                if (/trident/i.test(ua) || /msie/i.test(ua)) {
                    detectedBrowser = `Internet Explorer-${(/\brv[ :]+(\d+)/g.exec(ua) || [])[1] || 'Unk'}`;
                } else if (/Edg\//i.test(ua)) {
                    detectedBrowser = `Edge-${ua.match(/Edg\/(\d+)/)?.[1] || 'Unk'}`;
                } else if (/OPR\//i.test(ua) || /Opera/i.test(ua)) {
                    detectedBrowser = `Opera-${ua.match(/(?:OPR|Opera)\/(\d+)/)?.[1] || 'Unk'}`;
                } else if (/Firefox\//i.test(ua)) {
                    detectedBrowser = `Firefox-${ua.match(/Firefox\/(\d+)/)?.[1] || 'Unk'}`;
                } else if (/Chrome\//i.test(ua)) {
                    detectedBrowser = `Chrome-${ua.match(/Chrome\/(\d+)/)?.[1] || 'Unk'}`;
                } else if (/Safari\//i.test(ua) && !/Chrome\//i.test(ua)) {
                    detectedBrowser = `Safari-${ua.match(/Version\/(\d+\.\d+)/)?.[1] || 'Unk'}`;
                }
            }

            // If UA string pinpointed a cleaner version than generic "Chromium", overwrite it
            if (detectedBrowser !== 'Unk') {
                info.browser = detectedBrowser;
            }
        }

        // --- Browser fallback: detect crawlers, proxies, and minimal UAs ---
        if (info.browser === 'Unk') {
            if (/google\s*proxy|googlebot|google-chrome/i.test(ua)) {
                info.browser = 'GoogleBot-Proxy';
            } else if (/bot|crawler|spider|scrape/i.test(ua)) {
                info.browser = 'Bot-Crawler';
            } else if (/headless|phantom|puppeteer|playwright|selenium/i.test(ua)) {
                info.browser = 'Headless-Browser';
            } else if (info.device === 'Mobile' && info.isTouch && /linux/i.test(ua)) {
                info.browser = 'Mobile-WebKit';
            } else if (/linux|x11/i.test(ua)) {
                info.browser = 'Linux-Client';
            }
        }

        // --- OS detection: Client Hints first ---
        if (uaData && uaData.platform) {
            const p = uaData.platform;
            if (/android/i.test(p)) {
                info.os = `Android-${ua.match(/Android (\d+(?:\.\d+)?)/)?.[1] || 'Unk'}`;
            } else if (/windows/i.test(p)) {
                info.os = `Windows-${ua.match(/Windows NT (\d+\.\d+)/)?.[1] || 'Unk'}`;
            } else if (/macos/i.test(p)) {
                info.os = `macOS-${ua.match(/Mac OS X (\d+[_.\d]+)/)?.[1]?.replace(/_/g, '.') || 'Unk'}`;
            } else if (/ios/i.test(p)) {
                info.os = 'iOS';
            } else if (/linux/i.test(p)) {
                info.os = 'Linux-Unk';
            }
        }

        // --- OS detection: UA string fallback ---
        if (info.os === 'Unk') {
            if (/Windows NT/.test(ua)) {
                info.os = `Windows-${ua.match(/Windows NT (\d+\.\d+)/)?.[1] || 'Unk'}`;
            } else if (/Mac OS X/.test(ua)) {
                info.os = `macOS-${ua.match(/Mac OS X (\d+[_.\d]+)/)?.[1]?.replace(/_/g, '.') || 'Unk'}`;
            } else if (/Android/.test(ua)) {
                info.os = `Android-${ua.match(/Android (\d+(?:\.\d+)?)/)?.[1] || 'Unk'}`;
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
        }

        // --- OS fallback: infer from arch ---
        if (info.os === 'Unk' && /arm/i.test(arch)) {
            if (info.isTouch || /android|mobile/i.test(ua)) {
                info.os = 'Android-Unk';
            } else if (/armv[678]|aarch64/i.test(arch)) {
                info.os = 'Linux-ARM';
            } else {
                info.os = 'Android-Unk';
            }
        }

        // --- Device detection: Client Hints first ---
        if (uaData && uaData.mobile !== undefined) {
            info.device = uaData.mobile ? 'Mobile' : 'Desktop';
        }

        // --- Device detection: UA string fallback ---
        if (info.device === 'Unk') {
            if (/Mobi|iPhone|Android.+Mobile|Windows Phone/i.test(ua)) {
                info.device = 'Mobile';
            } else if (/iPad|Tablet|Nexus 7|SM-T|Kindle|Silk/i.test(ua)) {
                info.device = 'Tablet';
            } else if (/Windows|Macintosh|X11|Linux/i.test(ua)) {
                info.device = 'Desktop';
            }
        }

        // --- Device fallback: check screen size and touch capability ---
        if (info.device === 'Unk') {
            const screenWidth = window.screen.width;
            if (info.isTouch) {
                info.device = screenWidth <= 850 ? 'Mobile' : 'Tablet';
            } else if (screenWidth <= 1024) {
                info.device = 'Tablet';
            } else {
                info.device = 'Desktop';
            }
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
        { ip: '114.70.12.225', org: 'Ulsan National Institute of Science and Technology', os: 'Linux-Unk' }, // CANLab desktop
        { ip: '180.228.220.66', org: 'LG POWERCOMM', os: 'Linux-Unk' }, // Home laptop
        { ip: '180.228.220.66', org: 'LG POWERCOMM', os: 'Android-10' }, // Home mobile
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
